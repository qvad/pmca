#!/usr/bin/env python3
"""Comprehensive PMCA benchmark suite.

Tests code generation quality across difficulty levels with automated
validation probes that verify correctness without relying on LLM-generated tests.

Usage:
    python scripts/benchmark.py                     # all tasks, 7B config
    python scripts/benchmark.py config/test_14b.yaml  # all tasks, 14B config
    python scripts/benchmark.py --tier simple        # only simple tasks
    python scripts/benchmark.py --tier complex       # only complex tasks
    python scripts/benchmark.py --task calculator    # single task
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmca.models.config import Config
from pmca.orchestrator import Orchestrator
from pmca.utils.logger import setup_logging


# ---------------------------------------------------------------------------
# Validation probes — deterministic correctness checks run AFTER generation
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    """A single validation probe: a Python snippet that must exit 0."""
    name: str
    code: str  # Python code; can import from generated modules


@dataclass
class BenchTask:
    """A benchmark task definition."""
    name: str
    tier: str  # "simple", "medium", "complex"
    request: str  # The user request to PMCA
    probes: list[Probe] = field(default_factory=list)
    expected_difficulty: str = "complex"  # expected routing decision


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: list[BenchTask] = [
    # ── SIMPLE TIER ──────────────────────────────────────────────
    BenchTask(
        name="calculator",
        tier="simple",
        request=(
            "Create a Calculator class with add, subtract, multiply, divide methods. "
            "Each takes two numeric arguments and returns the result. "
            "divide(a, 0) should raise ValueError."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("add", "from calculator import Calculator; c = Calculator(); assert c.add(2, 3) == 5"),
            Probe("subtract", "from calculator import Calculator; c = Calculator(); assert c.subtract(10, 4) == 6"),
            Probe("multiply", "from calculator import Calculator; c = Calculator(); assert c.multiply(3, 7) == 21"),
            Probe("divide", "from calculator import Calculator; c = Calculator(); assert c.divide(15, 3) == 5.0"),
            Probe("divide_zero", """\
from calculator import Calculator
c = Calculator()
try:
    c.divide(5, 0)
    assert False, 'Should have raised ValueError'
except ValueError:
    pass
"""),
            Probe("negative", "from calculator import Calculator; c = Calculator(); assert c.add(-3, -7) == -10"),
        ],
    ),

    BenchTask(
        name="stack",
        tier="simple",
        request=(
            "Create a Stack class with push(item), pop() -> item, peek() -> item, "
            "is_empty() -> bool, size() -> int methods. "
            "pop() and peek() on empty stack should raise IndexError."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("push_pop", """\
from stack import Stack
s = Stack()
s.push(42)
assert s.pop() == 42
assert s.is_empty()
"""),
            Probe("peek", "from stack import Stack; s = Stack(); s.push(99); assert s.peek() == 99; assert s.size() == 1"),
            Probe("size", """\
from stack import Stack
s = Stack()
assert s.size() == 0
s.push(1); s.push(2); s.push(3)
assert s.size() == 3
"""),
            Probe("lifo_order", """\
from stack import Stack
s = Stack()
s.push('a'); s.push('b'); s.push('c')
assert s.pop() == 'c'
assert s.pop() == 'b'
assert s.pop() == 'a'
"""),
            Probe("empty_pop", """\
from stack import Stack
s = Stack()
try:
    s.pop()
    assert False, 'Should raise IndexError'
except IndexError:
    pass
"""),
            Probe("empty_peek", """\
from stack import Stack
s = Stack()
try:
    s.peek()
    assert False, 'Should raise IndexError'
except IndexError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="word_count",
        tier="simple",
        request=(
            "Create a function word_count(text: str) -> dict[str, int] that counts "
            "occurrences of each word in the text. Words should be lowercased. "
            "Return an empty dict for empty string."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("basic", "from word_count import word_count; assert word_count('hello world hello') == {'hello': 2, 'world': 1}"),
            Probe("empty", "from word_count import word_count; assert word_count('') == {}"),
            Probe("case", "from word_count import word_count; assert word_count('Cat cat CAT') == {'cat': 3}"),
            Probe("single", "from word_count import word_count; assert word_count('alone') == {'alone': 1}"),
        ],
    ),

    BenchTask(
        name="fizzbuzz",
        tier="simple",
        request=(
            "Create a function fizzbuzz(n: int) -> list[str] that returns a list of strings "
            "from 1 to n where: multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', "
            "multiples of both are 'FizzBuzz', and other numbers are their string representation."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("fifteen", """\
from fizzbuzz import fizzbuzz
result = fizzbuzz(15)
assert result[0] == '1'
assert result[2] == 'Fizz'
assert result[4] == 'Buzz'
assert result[14] == 'FizzBuzz'
assert len(result) == 15
"""),
            Probe("one", "from fizzbuzz import fizzbuzz; assert fizzbuzz(1) == ['1']"),
            Probe("three", "from fizzbuzz import fizzbuzz; assert fizzbuzz(3) == ['1', '2', 'Fizz']"),
        ],
    ),

    # ── MEDIUM TIER ──────────────────────────────────────────────
    BenchTask(
        name="text_stats",
        tier="medium",
        request=(
            "Create a TextStats class with methods: "
            "word_count(text) returns number of words (int), "
            "char_frequency(text) returns dict mapping each lowercase character to its count (skip spaces), "
            "most_common_word(text) returns the most frequent word (lowercased, return first alphabetically on tie), "
            "sentence_count(text) returns number of sentences (split on . ! ?). "
            "All methods should return 0, {}, '', or 0 respectively for empty string input."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("word_count", "from text_stats import TextStats; t = TextStats(); assert t.word_count('hello world foo') == 3"),
            Probe("word_count_empty", "from text_stats import TextStats; t = TextStats(); assert t.word_count('') == 0"),
            Probe("char_freq", """\
from text_stats import TextStats
t = TextStats()
freq = t.char_frequency('aab')
assert freq == {'a': 2, 'b': 1}
"""),
            Probe("char_freq_spaces", """\
from text_stats import TextStats
t = TextStats()
freq = t.char_frequency('a b a')
assert freq.get('a') == 2
assert freq.get('b') == 1
assert ' ' not in freq
"""),
            Probe("most_common", "from text_stats import TextStats; t = TextStats(); assert t.most_common_word('the cat and the dog') == 'the'"),
            Probe("sentence_count", "from text_stats import TextStats; t = TextStats(); assert t.sentence_count('Hello. World! Nice?') == 3"),
            Probe("sentence_empty", "from text_stats import TextStats; t = TextStats(); assert t.sentence_count('') == 0"),
        ],
    ),

    BenchTask(
        name="bank_account",
        tier="medium",
        request=(
            "Create a BankAccount class. Constructor takes owner (str) and optional initial_balance (float, default 0). "
            "Methods: deposit(amount) adds to balance, withdraw(amount) subtracts (raise ValueError if insufficient funds), "
            "get_balance() returns current balance, transfer(other_account, amount) moves money between accounts, "
            "get_history() returns list of strings describing each transaction (e.g. 'deposit 100.0', 'withdraw 50.0', "
            "'transfer out 25.0 to Bob', 'transfer in 25.0 from Alice')."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("deposit", """\
from bank_account import BankAccount
a = BankAccount('Alice', 100)
a.deposit(50)
assert a.get_balance() == 150
"""),
            Probe("withdraw", """\
from bank_account import BankAccount
a = BankAccount('Alice', 100)
a.withdraw(30)
assert a.get_balance() == 70
"""),
            Probe("withdraw_insufficient", """\
from bank_account import BankAccount
a = BankAccount('Alice', 50)
try:
    a.withdraw(100)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
assert a.get_balance() == 50
"""),
            Probe("transfer", """\
from bank_account import BankAccount
a = BankAccount('Alice', 200)
b = BankAccount('Bob', 50)
a.transfer(b, 75)
assert a.get_balance() == 125
assert b.get_balance() == 125
"""),
            Probe("history", """\
from bank_account import BankAccount
a = BankAccount('Alice', 0)
a.deposit(100)
a.withdraw(30)
h = a.get_history()
assert len(h) >= 2
assert 'deposit' in h[0].lower()
assert 'withdraw' in h[1].lower()
"""),
        ],
    ),

    BenchTask(
        name="linked_list",
        tier="medium",
        request=(
            "Create a LinkedList class (singly linked) with methods: "
            "append(value), prepend(value), delete(value) removes first occurrence (raise ValueError if not found), "
            "find(value) -> bool, size() -> int, to_list() -> list. "
            "The list should maintain insertion order."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("append_list", """\
from linked_list import LinkedList
ll = LinkedList()
ll.append(1); ll.append(2); ll.append(3)
assert ll.to_list() == [1, 2, 3]
"""),
            Probe("prepend", """\
from linked_list import LinkedList
ll = LinkedList()
ll.append(2); ll.prepend(1)
assert ll.to_list() == [1, 2]
"""),
            Probe("delete", """\
from linked_list import LinkedList
ll = LinkedList()
ll.append(1); ll.append(2); ll.append(3)
ll.delete(2)
assert ll.to_list() == [1, 3]
assert ll.size() == 2
"""),
            Probe("delete_not_found", """\
from linked_list import LinkedList
ll = LinkedList()
ll.append(1)
try:
    ll.delete(99)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
"""),
            Probe("find", """\
from linked_list import LinkedList
ll = LinkedList()
ll.append(10); ll.append(20)
assert ll.find(10) == True
assert ll.find(99) == False
"""),
            Probe("size", "from linked_list import LinkedList; ll = LinkedList(); assert ll.size() == 0; ll.append(1); assert ll.size() == 1"),
        ],
    ),

    # ── COMPLEX TIER ─────────────────────────────────────────────
    BenchTask(
        name="task_board",
        tier="complex",
        request=(
            "Create a TaskBoard class for managing tasks with priorities. "
            "Methods: add_task(title: str, priority: int) -> int (returns unique task_id starting from 1), "
            "complete_task(task_id: int) -> bool (returns True if found and completed, False if not found), "
            "get_pending() -> list[dict] returns pending tasks sorted by priority descending, each dict has keys 'id', 'title', 'priority', "
            "get_stats() -> dict with keys 'total', 'done', 'pending' (int values). "
            "Priority must be 1-5, raise ValueError for invalid priority."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_returns_id", """\
from task_board import TaskBoard
tb = TaskBoard()
id1 = tb.add_task('Task A', 3)
id2 = tb.add_task('Task B', 1)
assert id1 == 1
assert id2 == 2
"""),
            Probe("complete", """\
from task_board import TaskBoard
tb = TaskBoard()
tb.add_task('T1', 2)
assert tb.complete_task(1) == True
assert tb.complete_task(999) == False
"""),
            Probe("pending_sorted", """\
from task_board import TaskBoard
tb = TaskBoard()
tb.add_task('Low', 1)
tb.add_task('High', 5)
tb.add_task('Mid', 3)
pending = tb.get_pending()
assert len(pending) == 3
assert pending[0]['priority'] >= pending[1]['priority'] >= pending[2]['priority']
assert pending[0]['title'] == 'High'
"""),
            Probe("stats", """\
from task_board import TaskBoard
tb = TaskBoard()
tb.add_task('A', 1)
tb.add_task('B', 2)
tb.complete_task(1)
stats = tb.get_stats()
assert stats['total'] == 2
assert stats['done'] == 1
assert stats['pending'] == 1
"""),
            Probe("invalid_priority", """\
from task_board import TaskBoard
tb = TaskBoard()
try:
    tb.add_task('Bad', 0)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
try:
    tb.add_task('Bad', 6)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
"""),
            Probe("complete_removes_from_pending", """\
from task_board import TaskBoard
tb = TaskBoard()
tb.add_task('A', 3)
tb.add_task('B', 2)
tb.complete_task(1)
pending = tb.get_pending()
assert len(pending) == 1
assert pending[0]['title'] == 'B'
"""),
        ],
    ),

    BenchTask(
        name="matrix",
        tier="complex",
        request=(
            "Create a Matrix class that takes a 2D list of numbers in the constructor. "
            "Methods: rows() -> int, cols() -> int, get(row, col) -> number, "
            "transpose() -> Matrix (returns new Matrix), "
            "add(other) -> Matrix (element-wise addition, raise ValueError if dimensions differ), "
            "multiply(other) -> Matrix (matrix multiplication, raise ValueError if inner dimensions don't match), "
            "to_list() -> list[list] returns the underlying 2D list."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("dims", """\
from matrix import Matrix
m = Matrix([[1,2,3],[4,5,6]])
assert m.rows() == 2
assert m.cols() == 3
"""),
            Probe("get", "from matrix import Matrix; m = Matrix([[1,2],[3,4]]); assert m.get(0,0) == 1; assert m.get(1,1) == 4"),
            Probe("transpose", """\
from matrix import Matrix
m = Matrix([[1,2],[3,4]])
t = m.transpose()
assert t.to_list() == [[1,3],[2,4]]
"""),
            Probe("add", """\
from matrix import Matrix
a = Matrix([[1,2],[3,4]])
b = Matrix([[5,6],[7,8]])
c = a.add(b)
assert c.to_list() == [[6,8],[10,12]]
"""),
            Probe("add_mismatch", """\
from matrix import Matrix
a = Matrix([[1,2]])
b = Matrix([[1],[2]])
try:
    a.add(b)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
"""),
            Probe("multiply", """\
from matrix import Matrix
a = Matrix([[1,2],[3,4]])
b = Matrix([[5,6],[7,8]])
c = a.multiply(b)
assert c.to_list() == [[19,22],[43,50]]
"""),
            Probe("multiply_mismatch", """\
from matrix import Matrix
a = Matrix([[1,2,3]])
b = Matrix([[1,2,3]])
try:
    a.multiply(b)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="lru_cache",
        tier="complex",
        request=(
            "Create an LRUCache class with a capacity parameter. "
            "Methods: get(key) -> value or -1 if not found, "
            "put(key, value) inserts or updates, evicting least recently used if at capacity, "
            "size() -> int current number of items. "
            "Both get and put should count as 'use' for LRU ordering."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("basic_put_get", """\
from lru_cache import LRUCache
c = LRUCache(2)
c.put(1, 'a')
c.put(2, 'b')
assert c.get(1) == 'a'
assert c.get(2) == 'b'
"""),
            Probe("miss", "from lru_cache import LRUCache; c = LRUCache(2); assert c.get(99) == -1"),
            Probe("eviction", """\
from lru_cache import LRUCache
c = LRUCache(2)
c.put(1, 'a')
c.put(2, 'b')
c.put(3, 'c')  # evicts key 1
assert c.get(1) == -1
assert c.get(2) == 'b'
assert c.get(3) == 'c'
"""),
            Probe("lru_order", """\
from lru_cache import LRUCache
c = LRUCache(2)
c.put(1, 'a')
c.put(2, 'b')
c.get(1)        # use key 1, making key 2 the LRU
c.put(3, 'c')   # evicts key 2 (LRU)
assert c.get(1) == 'a'
assert c.get(2) == -1
assert c.get(3) == 'c'
"""),
            Probe("update", """\
from lru_cache import LRUCache
c = LRUCache(2)
c.put(1, 'a')
c.put(1, 'z')  # update
assert c.get(1) == 'z'
assert c.size() == 1
"""),
            Probe("size", """\
from lru_cache import LRUCache
c = LRUCache(3)
assert c.size() == 0
c.put(1, 'a'); c.put(2, 'b')
assert c.size() == 2
c.put(3, 'c'); c.put(4, 'd')  # evicts 1
assert c.size() == 3
"""),
        ],
    ),

    # ── COMPLEX TIER (continued) ────────────────────────────────────

    BenchTask(
        name="data_pipeline",
        tier="complex",
        request=(
            "Create a DataPipeline class for transforming list-of-dicts data. "
            "Constructor takes no arguments and initializes an empty pipeline. "
            "Methods (all return self for chaining): "
            "load(data: list[dict]) stores the data, "
            "filter(field: str, op: str, value) keeps rows where field matches "
            "(op is one of 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains'), "
            "transform(field: str, func_name: str) applies transformation to field in each row "
            "(func_name is one of 'upper', 'lower', 'strip', 'abs', 'int', 'float'), "
            "sort(field: str, reverse: bool = False) sorts rows by field, "
            "select(*fields) keeps only the specified fields in each row. "
            "execute() -> list[dict] runs the pipeline and returns results. "
            "aggregate(field: str, func: str) -> float|int computes a single value "
            "(func is one of 'sum', 'avg', 'min', 'max', 'count'). "
            "aggregate operates on current pipeline data (after filters etc). "
            "Raise KeyError if a field does not exist in any row. "
            "Raise ValueError if an unknown op, func_name, or func is given."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("load_execute", """\
from data_pipeline import DataPipeline
dp = DataPipeline()
data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
result = dp.load(data).execute()
assert result == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
"""),
            Probe("filter_gt", """\
from data_pipeline import DataPipeline
data = [{'x': 10}, {'x': 20}, {'x': 30}]
result = DataPipeline().load(data).filter('x', 'gt', 15).execute()
assert result == [{'x': 20}, {'x': 30}]
"""),
            Probe("filter_contains", """\
from data_pipeline import DataPipeline
data = [{'name': 'Alice'}, {'name': 'Bob'}, {'name': 'Alicia'}]
result = DataPipeline().load(data).filter('name', 'contains', 'Ali').execute()
assert result == [{'name': 'Alice'}, {'name': 'Alicia'}]
"""),
            Probe("transform_upper", """\
from data_pipeline import DataPipeline
data = [{'name': 'alice'}, {'name': 'bob'}]
result = DataPipeline().load(data).transform('name', 'upper').execute()
assert result == [{'name': 'ALICE'}, {'name': 'BOB'}]
"""),
            Probe("sort_reverse", """\
from data_pipeline import DataPipeline
data = [{'v': 3}, {'v': 1}, {'v': 2}]
result = DataPipeline().load(data).sort('v', reverse=True).execute()
assert [r['v'] for r in result] == [3, 2, 1]
"""),
            Probe("select_fields", """\
from data_pipeline import DataPipeline
data = [{'a': 1, 'b': 2, 'c': 3}]
result = DataPipeline().load(data).select('a', 'c').execute()
assert result == [{'a': 1, 'c': 3}]
"""),
            Probe("chaining", """\
from data_pipeline import DataPipeline
data = [
    {'name': 'alice', 'age': 30, 'score': 85},
    {'name': 'bob', 'age': 25, 'score': 90},
    {'name': 'carol', 'age': 35, 'score': 78},
    {'name': 'dave', 'age': 20, 'score': 95},
]
result = (DataPipeline()
    .load(data)
    .filter('age', 'gte', 25)
    .transform('name', 'upper')
    .sort('score', reverse=True)
    .select('name', 'score')
    .execute())
assert result == [
    {'name': 'BOB', 'score': 90},
    {'name': 'ALICE', 'score': 85},
    {'name': 'CAROL', 'score': 78},
]
"""),
            Probe("aggregate_sum_avg", """\
from data_pipeline import DataPipeline
data = [{'v': 10}, {'v': 20}, {'v': 30}]
dp = DataPipeline().load(data)
assert dp.aggregate('v', 'sum') == 60
assert dp.aggregate('v', 'avg') == 20.0
assert dp.aggregate('v', 'count') == 3
"""),
            Probe("empty_data", """\
from data_pipeline import DataPipeline
dp = DataPipeline().load([])
assert dp.execute() == []
assert dp.aggregate('x', 'count') == 0
"""),
            Probe("invalid_op", """\
from data_pipeline import DataPipeline
import traceback
data = [{'x': 1}]
try:
    DataPipeline().load(data).filter('x', 'bad_op', 1).execute()
    assert False, 'should have raised ValueError'
except ValueError:
    pass
try:
    DataPipeline().load(data).transform('x', 'bad_func').execute()
    assert False, 'should have raised ValueError'
except ValueError:
    pass
try:
    DataPipeline().load(data).aggregate('x', 'bad_agg')
    assert False, 'should have raised ValueError'
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="todo_manager",
        tier="complex",
        request=(
            "Create a TodoManager class for managing todo tasks. "
            "Constructor takes no arguments. "
            "Methods: "
            "add(title: str, due_date: str = None, priority: int = 1, tags: list = None) -> int "
            "adds a task and returns a unique integer id starting from 1. "
            "due_date is a string in 'YYYY-MM-DD' format or None. "
            "priority must be 1-5 (raise ValueError otherwise). "
            "tags is a list of strings or None (default to empty list). "
            "complete(task_id: int) -> bool marks task as done, returns False if id not found. "
            "delete(task_id: int) -> bool removes task, returns False if id not found. "
            "get(task_id: int) -> dict returns task as dict with keys: "
            "'id', 'title', 'done', 'due_date', 'priority', 'tags'. "
            "Raise KeyError if not found. "
            "list_tasks(status: str = 'all', sort_by: str = 'priority', tag: str = None) -> list[dict] "
            "status is 'all', 'done', or 'pending'. "
            "sort_by is 'priority' (descending) or 'due_date' (ascending, None dates last). "
            "tag filters to tasks containing that tag. "
            "search(query: str) -> list[dict] returns tasks where title contains query (case-insensitive). "
            "overdue(reference_date: str) -> list[dict] returns pending tasks with due_date before reference_date. "
            "stats() -> dict with keys 'total', 'done', 'pending' as integer counts."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_and_get", """\
from todo_manager import TodoManager
tm = TodoManager()
id1 = tm.add('Buy milk')
id2 = tm.add('Walk dog', due_date='2025-03-01', priority=3, tags=['pet'])
assert id1 == 1
assert id2 == 2
t = tm.get(id2)
assert t['title'] == 'Walk dog'
assert t['done'] is False
assert t['due_date'] == '2025-03-01'
assert t['priority'] == 3
assert t['tags'] == ['pet']
"""),
            Probe("complete_and_stats", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('A')
tm.add('B')
assert tm.complete(1) is True
assert tm.complete(999) is False
s = tm.stats()
assert s == {'total': 2, 'done': 1, 'pending': 1}
"""),
            Probe("delete", """\
from todo_manager import TodoManager
tm = TodoManager()
id1 = tm.add('A')
assert tm.delete(id1) is True
assert tm.delete(id1) is False
try:
    tm.get(id1)
    assert False, 'should raise KeyError'
except KeyError:
    pass
s = tm.stats()
assert s['total'] == 0
"""),
            Probe("list_by_priority", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('Low', priority=1)
tm.add('High', priority=5)
tm.add('Mid', priority=3)
tasks = tm.list_tasks(sort_by='priority')
priorities = [t['priority'] for t in tasks]
assert priorities == [5, 3, 1]
"""),
            Probe("list_filter_status", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('A')
tm.add('B')
tm.complete(1)
done = tm.list_tasks(status='done')
pending = tm.list_tasks(status='pending')
assert len(done) == 1 and done[0]['title'] == 'A'
assert len(pending) == 1 and pending[0]['title'] == 'B'
"""),
            Probe("list_filter_tag", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('Walk dog', tags=['pet', 'outdoor'])
tm.add('Feed cat', tags=['pet'])
tm.add('Buy milk', tags=['shopping'])
result = tm.list_tasks(tag='pet')
titles = sorted([t['title'] for t in result])
assert titles == ['Feed cat', 'Walk dog']
"""),
            Probe("search", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('Buy groceries')
tm.add('Buy new shoes')
tm.add('Walk the dog')
result = tm.search('buy')
assert len(result) == 2
titles = sorted([t['title'] for t in result])
assert titles == ['Buy groceries', 'Buy new shoes']
"""),
            Probe("overdue", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('Old task', due_date='2020-01-01')
tm.add('Future task', due_date='2099-12-31')
tm.add('No date task')
tm.add('Done old', due_date='2020-06-01')
tm.complete(4)
result = tm.overdue('2025-01-01')
assert len(result) == 1
assert result[0]['title'] == 'Old task'
"""),
            Probe("invalid_priority", """\
from todo_manager import TodoManager
tm = TodoManager()
try:
    tm.add('Bad', priority=0)
    assert False, 'should raise ValueError'
except ValueError:
    pass
try:
    tm.add('Bad', priority=6)
    assert False, 'should raise ValueError'
except ValueError:
    pass
id1 = tm.add('Good', priority=5)
assert id1 == 1
"""),
            Probe("sort_by_due_date", """\
from todo_manager import TodoManager
tm = TodoManager()
tm.add('No date', priority=1)
tm.add('Late', due_date='2025-12-01', priority=2)
tm.add('Early', due_date='2025-01-01', priority=3)
tasks = tm.list_tasks(sort_by='due_date')
titles = [t['title'] for t in tasks]
assert titles == ['Early', 'Late', 'No date']
"""),
        ],
    ),

    # ── EXTENDED BENCHMARK — 25 NEW TASKS ──────────────────────────────

    # ── SIMPLE TIER (5 new) ────────────────────────────────────────────

    BenchTask(
        name="temperature_converter",
        tier="simple",
        request=(
            "Create a TemperatureConverter class with methods: "
            "celsius_to_fahrenheit(c) returns float, "
            "fahrenheit_to_celsius(f) returns float, "
            "celsius_to_kelvin(c) returns float, "
            "kelvin_to_celsius(k) returns float. "
            "Raise ValueError if kelvin input is below 0 (absolute zero)."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("c_to_f", "from temperature_converter import TemperatureConverter; t = TemperatureConverter(); assert t.celsius_to_fahrenheit(0) == 32.0"),
            Probe("c_to_f_100", "from temperature_converter import TemperatureConverter; t = TemperatureConverter(); assert t.celsius_to_fahrenheit(100) == 212.0"),
            Probe("f_to_c", "from temperature_converter import TemperatureConverter; t = TemperatureConverter(); assert t.fahrenheit_to_celsius(32) == 0.0"),
            Probe("c_to_k", "from temperature_converter import TemperatureConverter; t = TemperatureConverter(); assert t.celsius_to_kelvin(0) == 273.15"),
            Probe("k_to_c", "from temperature_converter import TemperatureConverter; t = TemperatureConverter(); assert t.kelvin_to_celsius(273.15) == 0.0"),
            Probe("abs_zero", """\
from temperature_converter import TemperatureConverter
t = TemperatureConverter()
try:
    t.kelvin_to_celsius(-1)
    assert False, 'Should raise ValueError'
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="string_utils",
        tier="simple",
        request=(
            "Create a StringUtils class with methods: "
            "reverse(s: str) -> str, "
            "is_palindrome(s: str) -> bool (case-insensitive, ignoring spaces), "
            "count_vowels(s: str) -> int (a, e, i, o, u, case-insensitive), "
            "capitalize_words(s: str) -> str (capitalize first letter of each word)."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("reverse", "from string_utils import StringUtils; s = StringUtils(); assert s.reverse('hello') == 'olleh'"),
            Probe("palindrome_yes", "from string_utils import StringUtils; s = StringUtils(); assert s.is_palindrome('Racecar') is True"),
            Probe("palindrome_no", "from string_utils import StringUtils; s = StringUtils(); assert s.is_palindrome('hello') is False"),
            Probe("palindrome_spaces", "from string_utils import StringUtils; s = StringUtils(); assert s.is_palindrome('A man a plan a canal Panama') is True"),
            Probe("vowels", "from string_utils import StringUtils; s = StringUtils(); assert s.count_vowels('Hello World') == 3"),
            Probe("capitalize", "from string_utils import StringUtils; s = StringUtils(); assert s.capitalize_words('hello world foo') == 'Hello World Foo'"),
        ],
    ),

    BenchTask(
        name="queue_ds",
        tier="simple",
        request=(
            "Create a Queue class (FIFO) in a file called queue_ds.py. "
            "Methods: enqueue(item) adds to back, dequeue() removes and returns from front, "
            "peek() returns front item without removing, "
            "is_empty() -> bool, size() -> int. "
            "dequeue() and peek() on empty queue should raise IndexError."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("enqueue_dequeue", """\
from queue_ds import Queue
q = Queue()
q.enqueue(1); q.enqueue(2); q.enqueue(3)
assert q.dequeue() == 1
assert q.dequeue() == 2
"""),
            Probe("fifo_order", """\
from queue_ds import Queue
q = Queue()
q.enqueue('a'); q.enqueue('b'); q.enqueue('c')
assert q.dequeue() == 'a'
assert q.dequeue() == 'b'
assert q.dequeue() == 'c'
"""),
            Probe("peek", "from queue_ds import Queue; q = Queue(); q.enqueue(42); assert q.peek() == 42; assert q.size() == 1"),
            Probe("size_empty", "from queue_ds import Queue; q = Queue(); assert q.is_empty(); assert q.size() == 0"),
            Probe("empty_dequeue", """\
from queue_ds import Queue
q = Queue()
try:
    q.dequeue()
    assert False
except IndexError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="number_utils",
        tier="simple",
        request=(
            "Create the following functions: "
            "is_prime(n: int) -> bool returns True if n is prime (n < 2 returns False), "
            "factorial(n: int) -> int returns n! (raise ValueError for negative n), "
            "fibonacci(n: int) -> list[int] returns first n Fibonacci numbers starting [0, 1, 1, 2, ...], "
            "gcd(a: int, b: int) -> int returns greatest common divisor."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("prime_yes", "from number_utils import is_prime; assert is_prime(7) is True; assert is_prime(13) is True"),
            Probe("prime_no", "from number_utils import is_prime; assert is_prime(1) is False; assert is_prime(4) is False; assert is_prime(0) is False"),
            Probe("factorial", "from number_utils import factorial; assert factorial(5) == 120; assert factorial(0) == 1"),
            Probe("factorial_neg", """\
from number_utils import factorial
try:
    factorial(-1)
    assert False
except ValueError:
    pass
"""),
            Probe("fibonacci", "from number_utils import fibonacci; assert fibonacci(6) == [0, 1, 1, 2, 3, 5]"),
            Probe("gcd", "from number_utils import gcd; assert gcd(12, 8) == 4; assert gcd(17, 5) == 1"),
        ],
    ),

    BenchTask(
        name="shopping_cart",
        tier="simple",
        request=(
            "Create a ShoppingCart class. "
            "Methods: add_item(name: str, price: float, quantity: int = 1), "
            "remove_item(name: str) removes item (raise KeyError if not found), "
            "get_total() -> float returns sum of price * quantity for all items, "
            "item_count() -> int returns total number of items (sum of quantities), "
            "clear() removes all items."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("add_total", """\
from shopping_cart import ShoppingCart
c = ShoppingCart()
c.add_item('Apple', 1.50, 3)
c.add_item('Bread', 2.00)
assert c.get_total() == 6.50
"""),
            Probe("remove", """\
from shopping_cart import ShoppingCart
c = ShoppingCart()
c.add_item('Apple', 1.50)
c.remove_item('Apple')
assert c.get_total() == 0
"""),
            Probe("remove_missing", """\
from shopping_cart import ShoppingCart
c = ShoppingCart()
try:
    c.remove_item('Ghost')
    assert False
except KeyError:
    pass
"""),
            Probe("item_count", """\
from shopping_cart import ShoppingCart
c = ShoppingCart()
c.add_item('A', 1.0, 3)
c.add_item('B', 2.0, 2)
assert c.item_count() == 5
"""),
            Probe("clear", """\
from shopping_cart import ShoppingCart
c = ShoppingCart()
c.add_item('A', 1.0)
c.clear()
assert c.get_total() == 0
assert c.item_count() == 0
"""),
        ],
    ),

    # ── MEDIUM TIER (8 new) ────────────────────────────────────────────

    BenchTask(
        name="json_validator",
        tier="medium",
        request=(
            "Create a JsonValidator class. "
            "Methods: "
            "validate(schema: dict, data: dict) -> tuple[bool, list[str]] "
            "validates data against a schema and returns (is_valid, list_of_errors). "
            "Schema format: each key maps to a dict with 'type' (str, int, float, bool, list, dict), "
            "optional 'required' (bool, default True), optional 'min' / 'max' for numbers, "
            "optional 'min_length' / 'max_length' for strings. "
            "Return errors like 'field_name: expected type int, got str' or 'field_name: required field missing'."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("valid_data", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'name': {'type': 'str'}, 'age': {'type': 'int'}}
ok, errors = v.validate(schema, {'name': 'Alice', 'age': 30})
assert ok is True
assert errors == []
"""),
            Probe("wrong_type", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'age': {'type': 'int'}}
ok, errors = v.validate(schema, {'age': 'not_a_number'})
assert ok is False
assert len(errors) == 1
assert 'age' in errors[0]
"""),
            Probe("missing_required", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'name': {'type': 'str', 'required': True}}
ok, errors = v.validate(schema, {})
assert ok is False
assert any('required' in e.lower() or 'missing' in e.lower() for e in errors)
"""),
            Probe("optional_field", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'name': {'type': 'str'}, 'bio': {'type': 'str', 'required': False}}
ok, errors = v.validate(schema, {'name': 'Alice'})
assert ok is True
"""),
            Probe("min_max", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'age': {'type': 'int', 'min': 0, 'max': 150}}
ok1, _ = v.validate(schema, {'age': 25})
ok2, e2 = v.validate(schema, {'age': -1})
ok3, e3 = v.validate(schema, {'age': 200})
assert ok1 is True
assert ok2 is False
assert ok3 is False
"""),
            Probe("string_length", """\
from json_validator import JsonValidator
v = JsonValidator()
schema = {'code': {'type': 'str', 'min_length': 2, 'max_length': 5}}
ok1, _ = v.validate(schema, {'code': 'AB'})
ok2, _ = v.validate(schema, {'code': 'A'})
ok3, _ = v.validate(schema, {'code': 'ABCDEF'})
assert ok1 is True
assert ok2 is False
assert ok3 is False
"""),
        ],
    ),

    BenchTask(
        name="event_emitter",
        tier="medium",
        request=(
            "Create an EventEmitter class. "
            "Methods: on(event: str, callback: callable) registers a listener, "
            "off(event: str, callback: callable) removes a specific listener, "
            "emit(event: str, *args, **kwargs) calls all listeners for that event with the given args, "
            "once(event: str, callback: callable) registers a listener that fires only once, "
            "listener_count(event: str) -> int returns number of listeners for event."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("on_emit", """\
from event_emitter import EventEmitter
ee = EventEmitter()
results = []
ee.on('data', lambda x: results.append(x))
ee.emit('data', 42)
assert results == [42]
"""),
            Probe("multiple_listeners", """\
from event_emitter import EventEmitter
ee = EventEmitter()
r1, r2 = [], []
ee.on('data', lambda x: r1.append(x))
ee.on('data', lambda x: r2.append(x))
ee.emit('data', 'hello')
assert r1 == ['hello'] and r2 == ['hello']
"""),
            Probe("off", """\
from event_emitter import EventEmitter
ee = EventEmitter()
results = []
cb = lambda x: results.append(x)
ee.on('data', cb)
ee.off('data', cb)
ee.emit('data', 99)
assert results == []
"""),
            Probe("once", """\
from event_emitter import EventEmitter
ee = EventEmitter()
results = []
ee.once('click', lambda: results.append('clicked'))
ee.emit('click')
ee.emit('click')
assert results == ['clicked']
"""),
            Probe("listener_count", """\
from event_emitter import EventEmitter
ee = EventEmitter()
ee.on('a', lambda: None)
ee.on('a', lambda: None)
assert ee.listener_count('a') == 2
assert ee.listener_count('b') == 0
"""),
            Probe("no_event", """\
from event_emitter import EventEmitter
ee = EventEmitter()
ee.emit('nonexistent')  # should not raise
"""),
        ],
    ),

    BenchTask(
        name="rate_limiter",
        tier="medium",
        request=(
            "Create a RateLimiter class. Constructor takes max_requests (int) and "
            "window_seconds (float). "
            "Methods: allow(timestamp: float) -> bool returns True if the request is allowed "
            "(fewer than max_requests in the last window_seconds), False otherwise. "
            "get_usage(timestamp: float) -> dict returns {'allowed': int, 'remaining': int, 'reset_at': float} "
            "where reset_at is the timestamp when the oldest request in the window expires. "
            "reset() clears all tracked requests."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("allow_basic", """\
from rate_limiter import RateLimiter
rl = RateLimiter(3, 10.0)
assert rl.allow(1.0) is True
assert rl.allow(2.0) is True
assert rl.allow(3.0) is True
assert rl.allow(4.0) is False
"""),
            Probe("window_expiry", """\
from rate_limiter import RateLimiter
rl = RateLimiter(2, 5.0)
assert rl.allow(1.0) is True
assert rl.allow(2.0) is True
assert rl.allow(3.0) is False
assert rl.allow(7.0) is True  # 1.0 expired (7-5=2, > 1.0)
"""),
            Probe("usage", """\
from rate_limiter import RateLimiter
rl = RateLimiter(3, 10.0)
rl.allow(1.0)
rl.allow(5.0)
usage = rl.get_usage(6.0)
assert usage['allowed'] == 2
assert usage['remaining'] == 1
assert usage['reset_at'] == 11.0  # oldest(1.0) + window(10.0)
"""),
            Probe("reset", """\
from rate_limiter import RateLimiter
rl = RateLimiter(1, 60.0)
rl.allow(1.0)
assert rl.allow(2.0) is False
rl.reset()
assert rl.allow(3.0) is True
"""),
        ],
    ),

    BenchTask(
        name="interval_merger",
        tier="medium",
        request=(
            "Create a function merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] "
            "that takes a list of (start, end) intervals and merges overlapping ones. "
            "Return sorted list of merged intervals. "
            "Also create insert_interval(intervals: list[tuple[int, int]], new: tuple[int, int]) -> list[tuple[int, int]] "
            "that inserts a new interval into a sorted non-overlapping list and merges if needed. "
            "Also create is_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("merge_basic", """\
from interval_merger import merge_intervals
result = merge_intervals([(1,3), (2,6), (8,10), (15,18)])
assert result == [(1,6), (8,10), (15,18)]
"""),
            Probe("merge_all", """\
from interval_merger import merge_intervals
result = merge_intervals([(1,4), (2,5), (3,6)])
assert result == [(1,6)]
"""),
            Probe("merge_none", """\
from interval_merger import merge_intervals
result = merge_intervals([(1,2), (5,6), (9,10)])
assert result == [(1,2), (5,6), (9,10)]
"""),
            Probe("merge_empty", "from interval_merger import merge_intervals; assert merge_intervals([]) == []"),
            Probe("insert", """\
from interval_merger import insert_interval
result = insert_interval([(1,3), (6,9)], (2,5))
assert result == [(1,5), (6,9)]
"""),
            Probe("overlap", """\
from interval_merger import is_overlap
assert is_overlap((1,5), (3,7)) is True
assert is_overlap((1,3), (5,7)) is False
assert is_overlap((1,5), (5,7)) is True
"""),
        ],
    ),

    BenchTask(
        name="trie",
        tier="medium",
        request=(
            "Create a Trie class (prefix tree) for strings. "
            "Methods: insert(word: str), search(word: str) -> bool (exact match), "
            "starts_with(prefix: str) -> bool (any word starts with prefix), "
            "delete(word: str) -> bool (returns False if not found), "
            "words_with_prefix(prefix: str) -> list[str] returns all words with given prefix sorted alphabetically, "
            "size() -> int returns number of words stored."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("insert_search", """\
from trie import Trie
t = Trie()
t.insert('apple')
assert t.search('apple') is True
assert t.search('app') is False
"""),
            Probe("starts_with", """\
from trie import Trie
t = Trie()
t.insert('apple')
t.insert('application')
assert t.starts_with('app') is True
assert t.starts_with('xyz') is False
"""),
            Probe("prefix_words", """\
from trie import Trie
t = Trie()
t.insert('car'); t.insert('card'); t.insert('care'); t.insert('dog')
result = t.words_with_prefix('car')
assert result == ['car', 'card', 'care']
"""),
            Probe("delete", """\
from trie import Trie
t = Trie()
t.insert('hello')
assert t.delete('hello') is True
assert t.search('hello') is False
assert t.delete('hello') is False
"""),
            Probe("size", """\
from trie import Trie
t = Trie()
t.insert('a'); t.insert('b'); t.insert('c')
assert t.size() == 3
t.delete('b')
assert t.size() == 2
"""),
            Probe("delete_prefix_preserved", """\
from trie import Trie
t = Trie()
t.insert('app')
t.insert('apple')
t.delete('apple')
assert t.search('app') is True
assert t.search('apple') is False
"""),
        ],
    ),

    BenchTask(
        name="state_machine",
        tier="medium",
        request=(
            "Create a StateMachine class. Constructor takes initial_state (str). "
            "Methods: add_transition(from_state: str, event: str, to_state: str, action: callable = None) "
            "registers a transition. "
            "trigger(event: str) -> str transitions to new state if valid transition exists, "
            "calls action if provided, returns new state. Raise ValueError if no transition. "
            "current_state -> str property. "
            "history() -> list[tuple[str, str, str]] returns list of (from_state, event, to_state). "
            "can_trigger(event: str) -> bool returns whether event is valid from current state."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("basic_transition", """\
from state_machine import StateMachine
sm = StateMachine('idle')
sm.add_transition('idle', 'start', 'running')
result = sm.trigger('start')
assert result == 'running'
assert sm.current_state == 'running'
"""),
            Probe("invalid_event", """\
from state_machine import StateMachine
sm = StateMachine('idle')
try:
    sm.trigger('invalid')
    assert False
except ValueError:
    pass
"""),
            Probe("action_called", """\
from state_machine import StateMachine
log = []
sm = StateMachine('off')
sm.add_transition('off', 'press', 'on', action=lambda: log.append('turned_on'))
sm.trigger('press')
assert log == ['turned_on']
"""),
            Probe("history", """\
from state_machine import StateMachine
sm = StateMachine('a')
sm.add_transition('a', 'go', 'b')
sm.add_transition('b', 'go', 'c')
sm.trigger('go')
sm.trigger('go')
h = sm.history()
assert h == [('a', 'go', 'b'), ('b', 'go', 'c')]
"""),
            Probe("can_trigger", """\
from state_machine import StateMachine
sm = StateMachine('locked')
sm.add_transition('locked', 'unlock', 'unlocked')
assert sm.can_trigger('unlock') is True
assert sm.can_trigger('lock') is False
"""),
        ],
    ),

    BenchTask(
        name="expression_evaluator",
        tier="medium",
        request=(
            "Create a function evaluate(expression: str) -> float that evaluates "
            "simple arithmetic expressions with +, -, *, / operators and parentheses. "
            "Respect operator precedence (* and / before + and -). "
            "Handle spaces. Raise ValueError for invalid expressions (unbalanced parens, "
            "division by zero, invalid characters)."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("simple_add", "from expression_evaluator import evaluate; assert evaluate('2 + 3') == 5.0"),
            Probe("precedence", "from expression_evaluator import evaluate; assert evaluate('2 + 3 * 4') == 14.0"),
            Probe("parens", "from expression_evaluator import evaluate; assert evaluate('(2 + 3) * 4') == 20.0"),
            Probe("nested_parens", "from expression_evaluator import evaluate; assert evaluate('((1 + 2) * (3 + 4))') == 21.0"),
            Probe("division", "from expression_evaluator import evaluate; assert evaluate('10 / 4') == 2.5"),
            Probe("div_zero", """\
from expression_evaluator import evaluate
try:
    evaluate('5 / 0')
    assert False
except (ValueError, ZeroDivisionError):
    pass
"""),
            Probe("complex_expr", "from expression_evaluator import evaluate; assert abs(evaluate('3.5 * 2 + 1') - 8.0) < 0.001"),
        ],
    ),

    BenchTask(
        name="graph",
        tier="medium",
        request=(
            "Create a Graph class for an undirected, unweighted graph. "
            "Methods: add_vertex(v), add_edge(v1, v2), "
            "has_vertex(v) -> bool, has_edge(v1, v2) -> bool, "
            "neighbors(v) -> list (sorted), remove_edge(v1, v2), "
            "bfs(start, end) -> list returns shortest path as list of vertices (empty if no path), "
            "connected_components() -> list[list] returns list of components (each sorted, outer list sorted by first element)."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_vertex_edge", """\
from graph import Graph
g = Graph()
g.add_vertex('A'); g.add_vertex('B')
g.add_edge('A', 'B')
assert g.has_vertex('A')
assert g.has_edge('A', 'B')
assert g.has_edge('B', 'A')  # undirected
"""),
            Probe("neighbors", """\
from graph import Graph
g = Graph()
g.add_vertex('A'); g.add_vertex('B'); g.add_vertex('C')
g.add_edge('A', 'B'); g.add_edge('A', 'C')
assert g.neighbors('A') == ['B', 'C']
"""),
            Probe("bfs_path", """\
from graph import Graph
g = Graph()
for v in ['A','B','C','D']:
    g.add_vertex(v)
g.add_edge('A','B'); g.add_edge('B','C'); g.add_edge('C','D')
path = g.bfs('A', 'D')
assert path == ['A','B','C','D']
"""),
            Probe("bfs_no_path", """\
from graph import Graph
g = Graph()
g.add_vertex('A'); g.add_vertex('B')
path = g.bfs('A', 'B')
assert path == []
"""),
            Probe("components", """\
from graph import Graph
g = Graph()
for v in ['A','B','C','D','E']:
    g.add_vertex(v)
g.add_edge('A','B'); g.add_edge('D','E')
cc = g.connected_components()
assert len(cc) == 3  # {A,B}, {C}, {D,E}
"""),
            Probe("remove_edge", """\
from graph import Graph
g = Graph()
g.add_vertex('A'); g.add_vertex('B')
g.add_edge('A', 'B')
g.remove_edge('A', 'B')
assert g.has_edge('A', 'B') is False
"""),
        ],
    ),

    # ── COMPLEX TIER (12 new) ──────────────────────────────────────────

    BenchTask(
        name="mini_orm",
        tier="complex",
        request=(
            "Create a Table class that acts as a simple in-memory ORM. "
            "Constructor takes name (str) and columns (list of str). "
            "Methods: insert(row: dict) -> int inserts a row and returns auto-incremented id. "
            "All specified columns must be present in row (raise ValueError otherwise). "
            "select(where: dict = None) -> list[dict] returns rows matching all key=value pairs in where (None = all). "
            "update(row_id: int, values: dict) -> bool updates the row, returns False if not found. "
            "delete(row_id: int) -> bool deletes row, returns False if not found. "
            "count(where: dict = None) -> int returns count of matching rows. "
            "Each returned row dict includes an 'id' key."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("insert_select", """\
from mini_orm import Table
t = Table('users', ['name', 'age'])
id1 = t.insert({'name': 'Alice', 'age': 30})
id2 = t.insert({'name': 'Bob', 'age': 25})
assert id1 == 1 and id2 == 2
rows = t.select()
assert len(rows) == 2
assert rows[0]['name'] == 'Alice'
"""),
            Probe("select_where", """\
from mini_orm import Table
t = Table('users', ['name', 'age'])
t.insert({'name': 'Alice', 'age': 30})
t.insert({'name': 'Bob', 'age': 25})
t.insert({'name': 'Carol', 'age': 30})
result = t.select({'age': 30})
assert len(result) == 2
names = sorted(r['name'] for r in result)
assert names == ['Alice', 'Carol']
"""),
            Probe("update", """\
from mini_orm import Table
t = Table('users', ['name', 'age'])
t.insert({'name': 'Alice', 'age': 30})
assert t.update(1, {'age': 31}) is True
assert t.select({'id': 1})[0]['age'] == 31 if t.select({'id': 1}) else t.select()[0]['age'] == 31
"""),
            Probe("delete", """\
from mini_orm import Table
t = Table('users', ['name', 'age'])
t.insert({'name': 'Alice', 'age': 30})
assert t.delete(1) is True
assert t.delete(1) is False
assert t.count() == 0
"""),
            Probe("count", """\
from mini_orm import Table
t = Table('items', ['type'])
t.insert({'type': 'A'}); t.insert({'type': 'B'}); t.insert({'type': 'A'})
assert t.count() == 3
assert t.count({'type': 'A'}) == 2
"""),
            Probe("missing_column", """\
from mini_orm import Table
t = Table('users', ['name', 'age'])
try:
    t.insert({'name': 'Alice'})  # missing 'age'
    assert False
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="scheduler",
        tier="complex",
        request=(
            "Create a Scheduler class for scheduling non-overlapping time slots. "
            "Methods: book(start: int, end: int) -> bool books a slot if it doesn't overlap "
            "with existing bookings (returns True), otherwise returns False. start < end required. "
            "cancel(start: int, end: int) -> bool cancels exact slot (returns False if not found). "
            "get_bookings() -> list[tuple[int, int]] returns all bookings sorted by start time. "
            "is_available(start: int, end: int) -> bool checks if slot is free. "
            "next_available(after: int, duration: int) -> tuple[int, int] returns the next available "
            "slot of given duration starting from 'after'. "
            "Raise ValueError if start >= end."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("book_basic", """\
from scheduler import Scheduler
s = Scheduler()
assert s.book(9, 10) is True
assert s.book(10, 11) is True
assert s.book(9, 11) is False  # overlaps
"""),
            Probe("overlap_check", """\
from scheduler import Scheduler
s = Scheduler()
s.book(10, 20)
assert s.is_available(5, 10) is True
assert s.is_available(5, 15) is False
assert s.is_available(15, 25) is False
assert s.is_available(20, 25) is True
"""),
            Probe("cancel", """\
from scheduler import Scheduler
s = Scheduler()
s.book(9, 10)
assert s.cancel(9, 10) is True
assert s.cancel(9, 10) is False
assert s.book(9, 10) is True
"""),
            Probe("get_bookings", """\
from scheduler import Scheduler
s = Scheduler()
s.book(15, 20)
s.book(5, 10)
bookings = s.get_bookings()
assert bookings == [(5, 10), (15, 20)]
"""),
            Probe("next_available", """\
from scheduler import Scheduler
s = Scheduler()
s.book(10, 15)
s.book(20, 25)
slot = s.next_available(0, 5)
assert slot == (0, 5)
slot2 = s.next_available(10, 5)
assert slot2 == (15, 20)
slot3 = s.next_available(21, 5)
assert slot3 == (25, 30)
"""),
            Probe("invalid_range", """\
from scheduler import Scheduler
s = Scheduler()
try:
    s.book(10, 10)
    assert False
except ValueError:
    pass
try:
    s.book(10, 5)
    assert False
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="text_processor",
        tier="complex",
        request=(
            "Create a TextProcessor class. "
            "Methods: "
            "tokenize(text: str) -> list[str] splits text into words (lowercase, strip punctuation from edges). "
            "ngrams(text: str, n: int) -> list[tuple[str, ...]] returns n-grams from tokens. "
            "tf(text: str) -> dict[str, float] returns term frequency (count/total) for each token. "
            "similarity(text1: str, text2: str) -> float returns Jaccard similarity (intersection/union of token sets). "
            "summarize(text: str, n: int) -> str returns the n most frequent non-stopword tokens "
            "joined by space. Stopwords: a, an, the, is, in, on, at, to, and, of, for, it, this, that."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("tokenize", """\
from text_processor import TextProcessor
tp = TextProcessor()
tokens = tp.tokenize('Hello, World! Python is great.')
assert tokens == ['hello', 'world', 'python', 'is', 'great']
"""),
            Probe("ngrams", """\
from text_processor import TextProcessor
tp = TextProcessor()
result = tp.ngrams('one two three four', 2)
assert result == [('one', 'two'), ('two', 'three'), ('three', 'four')]
"""),
            Probe("tf", """\
from text_processor import TextProcessor
tp = TextProcessor()
result = tp.tf('the cat sat on the mat')
assert abs(result['the'] - 2/6) < 0.001
assert abs(result['cat'] - 1/6) < 0.001
"""),
            Probe("similarity", """\
from text_processor import TextProcessor
tp = TextProcessor()
sim = tp.similarity('the cat sat', 'the cat ran')
# intersection={the,cat}, union={the,cat,sat,ran} => 2/4 = 0.5
assert abs(sim - 0.5) < 0.001
"""),
            Probe("similarity_identical", """\
from text_processor import TextProcessor
tp = TextProcessor()
assert tp.similarity('hello world', 'hello world') == 1.0
assert tp.similarity('', '') == 0.0 or tp.similarity('', '') == 1.0  # edge case
"""),
            Probe("summarize", """\
from text_processor import TextProcessor
tp = TextProcessor()
text = 'the python python language is a great python language'
result = tp.summarize(text, 2)
# python(3) and language(2) are most frequent non-stopwords
words = result.split()
assert 'python' in words
assert 'language' in words
assert len(words) == 2
"""),
        ],
    ),

    BenchTask(
        name="binary_search_tree",
        tier="complex",
        request=(
            "Create a BST class (Binary Search Tree) for integers. "
            "Methods: insert(value: int), search(value: int) -> bool, "
            "delete(value: int) -> bool (returns False if not found), "
            "inorder() -> list[int] (sorted), "
            "min_value() -> int (raise ValueError if empty), "
            "max_value() -> int (raise ValueError if empty), "
            "height() -> int (empty tree has height 0, single node has height 1), "
            "size() -> int."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("insert_search", """\
from binary_search_tree import BST
tree = BST()
tree.insert(5); tree.insert(3); tree.insert(7)
assert tree.search(5) is True
assert tree.search(3) is True
assert tree.search(99) is False
"""),
            Probe("inorder", """\
from binary_search_tree import BST
tree = BST()
for v in [5, 3, 7, 1, 4]:
    tree.insert(v)
assert tree.inorder() == [1, 3, 4, 5, 7]
"""),
            Probe("delete_leaf", """\
from binary_search_tree import BST
tree = BST()
tree.insert(5); tree.insert(3); tree.insert(7)
assert tree.delete(3) is True
assert tree.search(3) is False
assert tree.inorder() == [5, 7]
"""),
            Probe("delete_two_children", """\
from binary_search_tree import BST
tree = BST()
for v in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(v)
tree.delete(5)  # root with two children
result = tree.inorder()
assert 5 not in result
assert sorted(result) == result  # still valid BST
assert len(result) == 6
"""),
            Probe("min_max", """\
from binary_search_tree import BST
tree = BST()
for v in [10, 5, 15, 3, 7]:
    tree.insert(v)
assert tree.min_value() == 3
assert tree.max_value() == 15
"""),
            Probe("height", """\
from binary_search_tree import BST
tree = BST()
assert tree.height() == 0
tree.insert(5)
assert tree.height() == 1
tree.insert(3); tree.insert(7)
assert tree.height() == 2
"""),
            Probe("size", """\
from binary_search_tree import BST
tree = BST()
assert tree.size() == 0
tree.insert(5); tree.insert(3); tree.insert(7)
assert tree.size() == 3
tree.delete(3)
assert tree.size() == 2
"""),
            Probe("empty_min", """\
from binary_search_tree import BST
tree = BST()
try:
    tree.min_value()
    assert False
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="inventory_system",
        tier="complex",
        request=(
            "Create an Inventory class for managing products. "
            "Methods: add_product(name: str, price: float, quantity: int, category: str) -> int "
            "returns auto-incremented product id. Raise ValueError if price < 0 or quantity < 0. "
            "restock(product_id: int, quantity: int) adds quantity. Raise KeyError if not found. "
            "sell(product_id: int, quantity: int) -> float subtracts quantity, returns total price. "
            "Raise KeyError if not found. Raise ValueError if insufficient stock. "
            "get_product(product_id: int) -> dict with keys 'id', 'name', 'price', 'quantity', 'category'. "
            "Raise KeyError if not found. "
            "search(query: str = None, category: str = None, in_stock: bool = None) -> list[dict] "
            "filters products. query matches name (case-insensitive substring). "
            "low_stock(threshold: int = 5) -> list[dict] returns products with quantity <= threshold "
            "sorted by quantity ascending. "
            "total_value() -> float returns sum of price * quantity for all products."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_get", """\
from inventory_system import Inventory
inv = Inventory()
id1 = inv.add_product('Widget', 9.99, 100, 'parts')
p = inv.get_product(id1)
assert p['name'] == 'Widget'
assert p['price'] == 9.99
assert p['quantity'] == 100
"""),
            Probe("sell", """\
from inventory_system import Inventory
inv = Inventory()
id1 = inv.add_product('Widget', 10.0, 50, 'parts')
total = inv.sell(id1, 3)
assert total == 30.0
assert inv.get_product(id1)['quantity'] == 47
"""),
            Probe("sell_insufficient", """\
from inventory_system import Inventory
inv = Inventory()
id1 = inv.add_product('Widget', 10.0, 2, 'parts')
try:
    inv.sell(id1, 5)
    assert False
except ValueError:
    pass
assert inv.get_product(id1)['quantity'] == 2  # unchanged
"""),
            Probe("restock", """\
from inventory_system import Inventory
inv = Inventory()
id1 = inv.add_product('Widget', 5.0, 10, 'parts')
inv.restock(id1, 20)
assert inv.get_product(id1)['quantity'] == 30
"""),
            Probe("search", """\
from inventory_system import Inventory
inv = Inventory()
inv.add_product('Blue Widget', 5.0, 10, 'parts')
inv.add_product('Red Widget', 6.0, 20, 'parts')
inv.add_product('Green Gadget', 15.0, 5, 'tools')
result = inv.search(query='widget')
assert len(result) == 2
result2 = inv.search(category='tools')
assert len(result2) == 1
"""),
            Probe("low_stock", """\
from inventory_system import Inventory
inv = Inventory()
inv.add_product('A', 1.0, 3, 'x')
inv.add_product('B', 1.0, 10, 'x')
inv.add_product('C', 1.0, 1, 'x')
low = inv.low_stock(5)
assert len(low) == 2
assert low[0]['name'] == 'C'  # quantity 1 first
assert low[1]['name'] == 'A'  # quantity 3 second
"""),
            Probe("total_value", """\
from inventory_system import Inventory
inv = Inventory()
inv.add_product('A', 10.0, 5, 'x')
inv.add_product('B', 20.0, 3, 'x')
assert inv.total_value() == 110.0
"""),
            Probe("invalid_price", """\
from inventory_system import Inventory
inv = Inventory()
try:
    inv.add_product('Bad', -5.0, 10, 'x')
    assert False
except ValueError:
    pass
"""),
        ],
    ),

    BenchTask(
        name="cache_system",
        tier="complex",
        request=(
            "Create a Cache class with TTL (time-to-live) support. "
            "Constructor takes default_ttl (float, seconds). "
            "Methods: set(key: str, value, ttl: float = None) stores value with expiry. "
            "If ttl is None, use default_ttl. "
            "get(key: str, current_time: float = None) -> value, returns None if expired or not found. "
            "If current_time not provided, use time.time(). "
            "delete(key: str) -> bool removes key, returns False if not found. "
            "clear() removes all entries. "
            "cleanup(current_time: float = None) removes all expired entries. "
            "size() -> int returns number of non-expired entries. "
            "keys() -> list[str] returns non-expired keys sorted."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("set_get", """\
from cache_system import Cache
import time
c = Cache(60.0)
now = time.time()
c.set('key1', 'value1')
assert c.get('key1', now) == 'value1'
"""),
            Probe("ttl_expiry", """\
from cache_system import Cache
c = Cache(10.0)
c.set('key1', 'val', ttl=5.0)
assert c.get('key1', current_time=100.0) == 'val'  # depends on when set
# Set with explicit timing
c2 = Cache(10.0)
c2.set('k', 'v', ttl=5.0)
import time
now = time.time()
assert c2.get('k', now + 1) == 'v'     # not expired
assert c2.get('k', now + 100) is None   # expired
"""),
            Probe("delete", """\
from cache_system import Cache
c = Cache(60.0)
c.set('key1', 'value1')
assert c.delete('key1') is True
assert c.get('key1') is None
assert c.delete('key1') is False
"""),
            Probe("clear_size", """\
from cache_system import Cache
c = Cache(60.0)
c.set('a', 1); c.set('b', 2); c.set('c', 3)
assert c.size() == 3
c.clear()
assert c.size() == 0
"""),
            Probe("keys_sorted", """\
from cache_system import Cache
c = Cache(60.0)
c.set('banana', 1); c.set('apple', 2); c.set('cherry', 3)
assert c.keys() == ['apple', 'banana', 'cherry']
"""),
        ],
    ),

    BenchTask(
        name="csv_parser",
        tier="complex",
        request=(
            "Create a CSVParser class. "
            "Methods: parse(text: str, delimiter: str = ',', has_header: bool = True) -> list[dict] "
            "parses CSV text. If has_header, first line is column names and returns list of dicts. "
            "If not has_header, returns list of lists. "
            "Handle quoted fields (double quotes around fields containing delimiter or newlines). "
            "to_csv(data: list[dict], delimiter: str = ',') -> str converts list of dicts back to CSV string "
            "with header row. "
            "filter_rows(data: list[dict], column: str, value) -> list[dict] returns rows where column equals value. "
            "sort_rows(data: list[dict], column: str, reverse: bool = False) -> list[dict] sorts by column."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("parse_basic", """\
from csv_parser import CSVParser
p = CSVParser()
text = 'name,age\\nAlice,30\\nBob,25'
result = p.parse(text)
assert len(result) == 2
assert result[0]['name'] == 'Alice'
assert result[0]['age'] == '30'
"""),
            Probe("parse_no_header", """\
from csv_parser import CSVParser
p = CSVParser()
text = 'a,b,c\\n1,2,3'
result = p.parse(text, has_header=False)
assert result == [['a','b','c'], ['1','2','3']]
"""),
            Probe("parse_quoted", """\
from csv_parser import CSVParser
p = CSVParser()
text = 'name,desc\\nAlice,"Hello, World"'
result = p.parse(text)
assert result[0]['desc'] == 'Hello, World'
"""),
            Probe("to_csv", """\
from csv_parser import CSVParser
p = CSVParser()
data = [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]
csv = p.to_csv(data)
lines = csv.strip().split('\\n')
assert len(lines) == 3  # header + 2 rows
assert 'name' in lines[0] and 'age' in lines[0]
"""),
            Probe("filter_rows", """\
from csv_parser import CSVParser
p = CSVParser()
data = [{'type': 'A', 'val': '1'}, {'type': 'B', 'val': '2'}, {'type': 'A', 'val': '3'}]
result = p.filter_rows(data, 'type', 'A')
assert len(result) == 2
"""),
            Probe("sort_rows", """\
from csv_parser import CSVParser
p = CSVParser()
data = [{'name': 'Charlie'}, {'name': 'Alice'}, {'name': 'Bob'}]
result = p.sort_rows(data, 'name')
assert [r['name'] for r in result] == ['Alice', 'Bob', 'Charlie']
"""),
        ],
    ),

    BenchTask(
        name="vector2d",
        tier="complex",
        request=(
            "Create a Vector2D class. Constructor takes x (float) and y (float). "
            "Properties: x, y (read-only). "
            "Methods: add(other) -> Vector2D, subtract(other) -> Vector2D, "
            "dot(other) -> float (dot product), "
            "magnitude() -> float (length), "
            "normalize() -> Vector2D (unit vector, raise ValueError if zero vector), "
            "rotate(angle_degrees: float) -> Vector2D (rotate counterclockwise), "
            "distance_to(other) -> float, "
            "angle_to(other) -> float (angle in degrees between two vectors), "
            "__eq__ for comparison (with floating point tolerance 1e-9), "
            "__repr__ returns 'Vector2D(x, y)'."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_sub", """\
from vector2d import Vector2D
a = Vector2D(1, 2)
b = Vector2D(3, 4)
c = a.add(b)
assert abs(c.x - 4) < 1e-9 and abs(c.y - 6) < 1e-9
d = a.subtract(b)
assert abs(d.x - (-2)) < 1e-9 and abs(d.y - (-2)) < 1e-9
"""),
            Probe("dot", "from vector2d import Vector2D; assert abs(Vector2D(1,0).dot(Vector2D(0,1))) < 1e-9"),
            Probe("magnitude", """\
from vector2d import Vector2D
import math
v = Vector2D(3, 4)
assert abs(v.magnitude() - 5.0) < 1e-9
"""),
            Probe("normalize", """\
from vector2d import Vector2D
v = Vector2D(3, 4)
n = v.normalize()
assert abs(n.magnitude() - 1.0) < 1e-9
"""),
            Probe("normalize_zero", """\
from vector2d import Vector2D
try:
    Vector2D(0, 0).normalize()
    assert False
except ValueError:
    pass
"""),
            Probe("rotate", """\
from vector2d import Vector2D
v = Vector2D(1, 0)
r = v.rotate(90)
assert abs(r.x) < 1e-6 and abs(r.y - 1) < 1e-6
"""),
            Probe("distance", """\
from vector2d import Vector2D
a = Vector2D(0, 0)
b = Vector2D(3, 4)
assert abs(a.distance_to(b) - 5.0) < 1e-9
"""),
        ],
    ),

    BenchTask(
        name="permission_system",
        tier="complex",
        request=(
            "Create a PermissionSystem class for role-based access control. "
            "Methods: create_role(role: str) creates a role (raise ValueError if exists). "
            "delete_role(role: str) removes role (raise KeyError if not found). "
            "grant(role: str, permission: str) adds permission to role. "
            "revoke(role: str, permission: str) removes permission (raise KeyError if not found). "
            "assign_role(user: str, role: str) assigns role to user. "
            "unassign_role(user: str, role: str) removes role from user. "
            "has_permission(user: str, permission: str) -> bool checks if any of user's roles has permission. "
            "get_permissions(user: str) -> set[str] returns all permissions across all roles. "
            "get_users_with_permission(permission: str) -> list[str] returns sorted list of users."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("create_grant", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
ps.create_role('admin')
ps.grant('admin', 'read')
ps.grant('admin', 'write')
ps.assign_role('alice', 'admin')
assert ps.has_permission('alice', 'read') is True
assert ps.has_permission('alice', 'delete') is False
"""),
            Probe("multi_role", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
ps.create_role('viewer')
ps.create_role('editor')
ps.grant('viewer', 'read')
ps.grant('editor', 'write')
ps.assign_role('bob', 'viewer')
ps.assign_role('bob', 'editor')
perms = ps.get_permissions('bob')
assert 'read' in perms and 'write' in perms
"""),
            Probe("revoke", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
ps.create_role('admin')
ps.grant('admin', 'delete')
ps.assign_role('alice', 'admin')
ps.revoke('admin', 'delete')
assert ps.has_permission('alice', 'delete') is False
"""),
            Probe("users_with_perm", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
ps.create_role('admin')
ps.grant('admin', 'write')
ps.assign_role('bob', 'admin')
ps.assign_role('alice', 'admin')
users = ps.get_users_with_permission('write')
assert users == ['alice', 'bob']
"""),
            Probe("duplicate_role", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
ps.create_role('admin')
try:
    ps.create_role('admin')
    assert False
except ValueError:
    pass
"""),
            Probe("no_role_user", """\
from permission_system import PermissionSystem
ps = PermissionSystem()
assert ps.has_permission('nobody', 'anything') is False
assert ps.get_permissions('nobody') == set()
"""),
        ],
    ),

    BenchTask(
        name="sparse_matrix",
        tier="complex",
        request=(
            "Create a SparseMatrix class for efficient storage of matrices with many zeros. "
            "Constructor takes rows (int) and cols (int). "
            "Methods: set(row: int, col: int, value: float) sets value (remove if zero). "
            "get(row: int, col: int) -> float returns value (0.0 if not stored). "
            "Raise IndexError if row/col out of bounds. "
            "nnz() -> int returns number of non-zero elements. "
            "add(other: SparseMatrix) -> SparseMatrix element-wise addition (raise ValueError if dimensions differ). "
            "multiply(other: SparseMatrix) -> SparseMatrix matrix multiplication (raise ValueError if incompatible). "
            "transpose() -> SparseMatrix. "
            "to_dense() -> list[list[float]] returns full 2D list."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("set_get", """\
from sparse_matrix import SparseMatrix
m = SparseMatrix(3, 3)
m.set(0, 0, 5.0)
m.set(1, 2, 3.0)
assert m.get(0, 0) == 5.0
assert m.get(1, 1) == 0.0
assert m.get(1, 2) == 3.0
"""),
            Probe("nnz", """\
from sparse_matrix import SparseMatrix
m = SparseMatrix(3, 3)
m.set(0, 0, 1.0)
m.set(1, 1, 2.0)
m.set(2, 2, 3.0)
assert m.nnz() == 3
m.set(1, 1, 0.0)  # remove
assert m.nnz() == 2
"""),
            Probe("bounds", """\
from sparse_matrix import SparseMatrix
m = SparseMatrix(2, 2)
try:
    m.get(5, 0)
    assert False
except IndexError:
    pass
"""),
            Probe("add", """\
from sparse_matrix import SparseMatrix
a = SparseMatrix(2, 2)
a.set(0, 0, 1.0); a.set(1, 1, 2.0)
b = SparseMatrix(2, 2)
b.set(0, 0, 3.0); b.set(0, 1, 4.0)
c = a.add(b)
assert c.get(0, 0) == 4.0
assert c.get(0, 1) == 4.0
assert c.get(1, 1) == 2.0
"""),
            Probe("transpose", """\
from sparse_matrix import SparseMatrix
m = SparseMatrix(2, 3)
m.set(0, 2, 5.0)
t = m.transpose()
assert t.get(2, 0) == 5.0
assert t.get(0, 2) == 0.0
"""),
            Probe("to_dense", """\
from sparse_matrix import SparseMatrix
m = SparseMatrix(2, 2)
m.set(0, 0, 1.0)
m.set(1, 1, 2.0)
assert m.to_dense() == [[1.0, 0.0], [0.0, 2.0]]
"""),
            Probe("multiply", """\
from sparse_matrix import SparseMatrix
a = SparseMatrix(2, 2)
a.set(0, 0, 1.0); a.set(0, 1, 2.0)
a.set(1, 0, 3.0); a.set(1, 1, 4.0)
b = SparseMatrix(2, 1)
b.set(0, 0, 5.0); b.set(1, 0, 6.0)
c = a.multiply(b)
assert c.get(0, 0) == 17.0  # 1*5 + 2*6
assert c.get(1, 0) == 39.0  # 3*5 + 4*6
"""),
        ],
    ),

    BenchTask(
        name="config_parser",
        tier="complex",
        request=(
            "Create a ConfigParser class for parsing INI-style configuration. "
            "Methods: parse(text: str) parses config text. "
            "Sections are [section_name], key-value pairs are key = value. "
            "Lines starting with # or ; are comments. Empty lines are ignored. "
            "get(section: str, key: str, default=None) returns value as string, or default. "
            "get_int(section: str, key: str, default: int = 0) -> int. "
            "get_bool(section: str, key: str, default: bool = False) -> bool "
            "(true/yes/1/on are True, false/no/0/off are False, case-insensitive). "
            "sections() -> list[str] returns sorted section names. "
            "items(section: str) -> dict[str, str] returns all key-value pairs in section. "
            "has_section(section: str) -> bool. "
            "set(section: str, key: str, value: str) sets a value (creates section if needed). "
            "to_string() -> str serializes back to INI format."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("parse_get", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('[database]\\nhost = localhost\\nport = 5432')
assert cp.get('database', 'host') == 'localhost'
assert cp.get('database', 'port') == '5432'
"""),
            Probe("get_int", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('[db]\\nport = 3306')
assert cp.get_int('db', 'port') == 3306
assert cp.get_int('db', 'timeout', default=30) == 30
"""),
            Probe("get_bool", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('[app]\\ndebug = true\\nverbose = no\\nenabled = 1')
assert cp.get_bool('app', 'debug') is True
assert cp.get_bool('app', 'verbose') is False
assert cp.get_bool('app', 'enabled') is True
"""),
            Probe("comments_ignored", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('# comment\\n[section]\\n; another comment\\nkey = value')
assert cp.get('section', 'key') == 'value'
assert cp.has_section('section')
"""),
            Probe("sections", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('[beta]\\nx=1\\n[alpha]\\ny=2')
assert cp.sections() == ['alpha', 'beta']
"""),
            Probe("set_serialize", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.set('new_section', 'key1', 'val1')
assert cp.get('new_section', 'key1') == 'val1'
output = cp.to_string()
assert 'new_section' in output
assert 'key1' in output
"""),
            Probe("items", """\
from config_parser import ConfigParser
cp = ConfigParser()
cp.parse('[db]\\nhost = localhost\\nport = 5432')
items = cp.items('db')
assert items == {'host': 'localhost', 'port': '5432'}
"""),
        ],
    ),

    BenchTask(
        name="task_scheduler",
        tier="complex",
        request=(
            "Create a TaskScheduler class for scheduling tasks with dependencies. "
            "Methods: add_task(name: str, duration: int, dependencies: list[str] = None) "
            "adds a task. Raise ValueError if duplicate name. "
            "get_execution_order() -> list[str] returns topological sort order. "
            "Raise ValueError if circular dependency detected. "
            "get_critical_path() -> tuple[list[str], int] returns the longest path of tasks "
            "and total duration (sum of durations on the path). "
            "can_run(task_name: str) -> bool returns True if all dependencies are marked complete. "
            "complete(task_name: str) marks task as complete. Raise KeyError if not found. "
            "get_ready_tasks() -> list[str] returns sorted list of tasks whose dependencies are all complete "
            "and that are not yet complete themselves."
        ),
        expected_difficulty="complex",
        probes=[
            Probe("add_and_order", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('build', 10, ['compile'])
ts.add_task('compile', 5)
ts.add_task('test', 3, ['build'])
order = ts.get_execution_order()
assert order.index('compile') < order.index('build')
assert order.index('build') < order.index('test')
"""),
            Probe("circular_dep", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('A', 1, ['B'])
ts.add_task('B', 1, ['A'])
try:
    ts.get_execution_order()
    assert False
except ValueError:
    pass
"""),
            Probe("critical_path", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('A', 3)
ts.add_task('B', 5, ['A'])
ts.add_task('C', 2, ['A'])
ts.add_task('D', 4, ['B', 'C'])
path, duration = ts.get_critical_path()
assert duration == 12  # A(3) -> B(5) -> D(4) = 12
assert 'A' in path and 'B' in path and 'D' in path
"""),
            Probe("ready_tasks", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('A', 1)
ts.add_task('B', 1)
ts.add_task('C', 1, ['A'])
ready = ts.get_ready_tasks()
assert sorted(ready) == ['A', 'B']
ts.complete('A')
ready2 = ts.get_ready_tasks()
assert sorted(ready2) == ['B', 'C']
"""),
            Probe("can_run", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('X', 1)
ts.add_task('Y', 1, ['X'])
assert ts.can_run('X') is True
assert ts.can_run('Y') is False
ts.complete('X')
assert ts.can_run('Y') is True
"""),
            Probe("duplicate_task", """\
from task_scheduler import TaskScheduler
ts = TaskScheduler()
ts.add_task('A', 1)
try:
    ts.add_task('A', 2)
    assert False
except ValueError:
    pass
"""),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_name: str
    tier: str
    status: str
    passed: bool
    elapsed_s: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_duration_ms: float = 0
    retries: int = 0
    gate_stats: dict = field(default_factory=dict)
    probes_total: int = 0
    probes_passed: int = 0
    probe_details: list[tuple[str, bool, str]] = field(default_factory=list)
    error: str = ""


def _discover_module_name(workspace: Path, expected_name: str) -> str | None:
    """Find the actual module name in src/ for an expected name.

    The 7B model often generates file names like 'calculator_class_add.py'
    instead of 'calculator.py'. This function finds the actual file.
    """
    src_dir = workspace / "src"
    if not src_dir.is_dir():
        return None
    # Exact match
    if (src_dir / f"{expected_name}.py").exists():
        return expected_name
    # Fuzzy: find files containing the expected name
    expected_norm = expected_name.replace("_", "")
    for f in src_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        stem_norm = f.stem.replace("_", "")
        if expected_name in f.stem or f.stem in expected_name:
            return f.stem
        # Normalize underscores: lru_cache matches lrucache
        if expected_norm in stem_norm or stem_norm in expected_norm:
            return f.stem
    # Last resort: if only one non-init .py file exists, use it
    py_files = [f for f in src_dir.glob("*.py") if f.name != "__init__.py"]
    if len(py_files) == 1:
        return py_files[0].stem
    return None


def run_probes(workspace: Path, probes: list[Probe], python_exe: str) -> list[tuple[str, bool, str]]:
    """Run validation probes against generated code. Returns (name, passed, error)."""
    ws = workspace.resolve()

    # Build module name mapping for probe import rewriting
    # Probes use expected names like "from calculator import..."
    # but PMCA may generate "src/calculator_class_add.py"
    import re as _re
    module_map: dict[str, str] = {}
    for probe in probes:
        for m in _re.finditer(r"from\s+(\w+)\s+import", probe.code):
            expected = m.group(1)
            if expected not in module_map:
                actual = _discover_module_name(ws, expected)
                if actual and actual != expected:
                    module_map[expected] = actual

    results = []
    for probe in probes:
        code = probe.code
        # Rewrite imports to use actual module names
        for expected, actual in module_map.items():
            code = code.replace(f"from {expected} import", f"from {actual} import")
        try:
            proc = subprocess.run(
                [python_exe, "-c", code],
                capture_output=True, text=True,
                cwd=str(ws),
                env={
                    **__import__("os").environ,
                    "PYTHONPATH": f"{ws}:{ws / 'src'}",
                },
                timeout=10,
            )
            if proc.returncode == 0:
                results.append((probe.name, True, ""))
            else:
                err = (proc.stderr or proc.stdout).strip().split("\n")[-1]
                results.append((probe.name, False, err))
        except subprocess.TimeoutExpired:
            results.append((probe.name, False, "timeout"))
        except Exception as e:
            results.append((probe.name, False, str(e)))
    return results


async def run_task(config: Config, task: BenchTask, python_exe: str, think: bool | None = None) -> TaskResult:
    """Run a single benchmark task through PMCA and validate with probes."""
    workspace = Path(f"./workspace/bench_{task.name}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(config, workspace)

    t0 = time.monotonic()
    try:
        root = await orch.run(task.request, think=think)
        elapsed = time.monotonic() - t0

        mm = orch._model_manager
        result = TaskResult(
            task_name=task.name,
            tier=task.tier,
            status=root.status.value,
            passed=root.is_complete,
            elapsed_s=round(elapsed, 1),
            llm_calls=mm.total_llm_calls,
            prompt_tokens=mm.total_prompt_tokens,
            completion_tokens=mm.total_completion_tokens,
            llm_duration_ms=round(mm.total_llm_duration_ms, 0),
            retries=root.retry_count,
            gate_stats=dict(orch._gate_stats),
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        result = TaskResult(
            task_name=task.name,
            tier=task.tier,
            status="error",
            passed=False,
            elapsed_s=round(elapsed, 1),
            error=str(exc),
        )

    # Run validation probes regardless of PMCA status
    probe_results = run_probes(workspace, task.probes, python_exe)
    result.probes_total = len(probe_results)
    result.probes_passed = sum(1 for _, ok, _ in probe_results if ok)
    result.probe_details = probe_results

    return result


async def run_benchmark(config_path: str, tasks: list[BenchTask], think: bool | None = None) -> list[TaskResult]:
    """Run all selected benchmark tasks."""
    config = Config.from_yaml(Path(config_path))
    python_exe = sys.executable
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(tasks)}] {task.name} (tier={task.tier})")
        print(f"{'='*70}")

        result = await run_task(config, task, python_exe, think=think)
        results.append(result)

        # Print inline result
        p_str = f"{result.probes_passed}/{result.probes_total} probes"
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  -> {status} | {p_str} | {result.llm_calls} calls | {result.elapsed_s}s")
        for name, ok, err in result.probe_details:
            mark = "+" if ok else "X"
            print(f"     [{mark}] {name}" + (f" — {err}" if err else ""))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: list[TaskResult]) -> None:
    total = len(results)
    pmca_passed = sum(1 for r in results if r.passed)
    all_probes = sum(r.probes_total for r in results)
    passed_probes = sum(r.probes_passed for r in results)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    total_calls = sum(r.llm_calls for r in results)
    total_time = sum(r.elapsed_s for r in results)

    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"PMCA tasks:    {pmca_passed}/{total} passed")
    print(f"Probes:        {passed_probes}/{all_probes} passed ({passed_probes/max(all_probes,1)*100:.0f}%)")
    print(f"LLM calls:     {total_calls}")
    print(f"Total tokens:  {total_tokens:,}")
    print(f"Total time:    {total_time:.1f}s")

    # Per-tier breakdown
    for tier in ("simple", "medium", "complex"):
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue
        tp = sum(1 for r in tier_results if r.passed)
        pp = sum(r.probes_passed for r in tier_results)
        pt = sum(r.probes_total for r in tier_results)
        print(f"\n  {tier.upper()}: {tp}/{len(tier_results)} tasks, {pp}/{pt} probes")

    # Per-task table
    print(f"\n{'Task':<16} {'Tier':<8} {'PMCA':<6} {'Probes':<10} {'Calls':<6} {'Tokens':<10} {'Time':<8} {'Retries':<4}")
    print("-" * 76)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        probes = f"{r.probes_passed}/{r.probes_total}"
        tokens = r.prompt_tokens + r.completion_tokens
        print(f"{r.task_name:<16} {r.tier:<8} {status:<6} {probes:<10} {r.llm_calls:<6} {tokens:<10,} {r.elapsed_s:<8} {r.retries}")

    # Failed probes detail
    failed = [(r.task_name, name, err)
              for r in results for name, ok, err in r.probe_details if not ok]
    if failed:
        print(f"\nFailed probes ({len(failed)}):")
        for task, probe, err in failed:
            print(f"  {task}/{probe}: {err[:80]}")


def save_results(results: list[TaskResult], path: str) -> None:
    """Save results as JSON for tracking over time."""
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {
                "task": r.task_name,
                "tier": r.tier,
                "passed": r.passed,
                "probes_passed": r.probes_passed,
                "probes_total": r.probes_total,
                "llm_calls": r.llm_calls,
                "tokens": r.prompt_tokens + r.completion_tokens,
                "elapsed_s": r.elapsed_s,
                "retries": r.retries,
                "gate_stats": r.gate_stats,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PMCA Benchmark Suite")
    parser.add_argument("config", nargs="?", default="config/test_7b.yaml",
                        help="Path to PMCA config YAML")
    parser.add_argument("--tier", choices=["simple", "medium", "complex"],
                        help="Only run tasks from this tier")
    parser.add_argument("--task", help="Only run a specific task by name")
    parser.add_argument("--output", "-o", default="./benchmark_results.json",
                        help="Save results JSON to this path")
    parser.add_argument("--think", choices=["true", "false"],
                        help="Override think mode (true/false) for reasoning models")
    args = parser.parse_args()

    setup_logging("INFO", "./benchmark.log")

    # Filter tasks
    tasks = TASKS
    if args.tier:
        tasks = [t for t in tasks if t.tier == args.tier]
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]

    if not tasks:
        print("No tasks match the filter")
        sys.exit(1)

    print(f"Config: {args.config}")
    print(f"Tasks:  {len(tasks)} ({', '.join(t.name for t in tasks)})")

    think = None
    if args.think == "true":
        think = True
    elif args.think == "false":
        think = False

    results = asyncio.run(run_benchmark(args.config, tasks, think=think))
    print_summary(results)
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
