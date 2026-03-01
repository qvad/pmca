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
        name="counter",
        tier="simple",
        request=(
            "Create a function word_count(text: str) -> dict[str, int] that counts "
            "occurrences of each word in the text. Words should be lowercased. "
            "Return an empty dict for empty string."
        ),
        expected_difficulty="simple",
        probes=[
            Probe("basic", "from counter import word_count; assert word_count('hello world hello') == {'hello': 2, 'world': 1}"),
            Probe("empty", "from counter import word_count; assert word_count('') == {}"),
            Probe("case", "from counter import word_count; assert word_count('Cat cat CAT') == {'cat': 3}"),
            Probe("single", "from counter import word_count; assert word_count('alone') == {'alone': 1}"),
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
    for f in src_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        if expected_name in f.stem or f.stem in expected_name:
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


async def run_task(config: Config, task: BenchTask, python_exe: str) -> TaskResult:
    """Run a single benchmark task through PMCA and validate with probes."""
    workspace = Path(f"./workspace/bench_{task.name}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator(config, workspace)

    t0 = time.monotonic()
    try:
        root = await orch.run(task.request)
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


async def run_benchmark(config_path: str, tasks: list[BenchTask]) -> list[TaskResult]:
    """Run all selected benchmark tasks."""
    config = Config.from_yaml(Path(config_path))
    python_exe = sys.executable
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(tasks)}] {task.name} (tier={task.tier})")
        print(f"{'='*70}")

        result = await run_task(config, task, python_exe)
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

    results = asyncio.run(run_benchmark(args.config, tasks))
    print_summary(results)
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
