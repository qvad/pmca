"""Language-specific coding skills for all supported languages.

Injected into the coder system prompt based on detected language.
Covers the top 20 languages with idiomatic patterns and conventions.
"""

RUST_SKILLS = """
## Rust Coding Standards:
- Use Result<T, E> for fallible operations, never panic in library code
- Prefer &str over String for function parameters, return String when ownership transfers
- Use derive macros: #[derive(Debug, Clone, PartialEq)] on data types
- Error handling: implement std::error::Error, use thiserror or anyhow
- Use iterators and closures over manual loops: .iter().map().filter().collect()
- Ownership: prefer borrowing (&T) over cloning. Move semantics by default.
- Tests go in the same file: #[cfg(test)] mod tests { use super::*; }
- Naming: snake_case for functions/variables, PascalCase for types/traits
"""

JAVA_SKILLS = """
## Java Coding Standards:
- Use records for immutable data classes (Java 16+)
- Prefer List.of(), Map.of() for immutable collections
- Use Optional<T> instead of null returns
- Exception handling: checked exceptions for recoverable, unchecked for bugs
- Follow SOLID principles, prefer composition over inheritance
- Use try-with-resources for AutoCloseable objects
- Naming: camelCase for methods/variables, PascalCase for classes, SCREAMING_CASE for constants
- Tests: JUnit 5 with @Test, @BeforeEach, assertThat() from AssertJ
"""

KOTLIN_SKILLS = """
## Kotlin Coding Standards:
- Use data class for value objects, sealed class for algebraic types
- Null safety: use ?.let{}, ?:, !! only when certain
- Use when expression instead of if/else chains
- Extension functions for utility methods
- Coroutines: suspend fun for async, use structured concurrency
- Naming: camelCase for functions, PascalCase for classes
- Tests: JUnit 5 or kotest, use shouldBe matcher style
"""

CSHARP_SKILLS = """
## C# Coding Standards:
- Use record types for immutable data (C# 9+)
- Pattern matching with switch expressions
- Async/await for I/O operations, never .Result or .Wait()
- Use nullable reference types (enable in project)
- LINQ for collection operations
- Dependency injection via constructor
- Naming: PascalCase for public members, _camelCase for private fields
- Tests: xUnit with [Fact] and [Theory], FluentAssertions
"""

CPP_SKILLS = """
## C++ Coding Standards:
- Use smart pointers: std::unique_ptr, std::shared_ptr — never raw new/delete
- Prefer std::string_view over const std::string& for parameters
- Use auto for complex types, explicit types for clarity
- RAII: resources acquired in constructor, released in destructor
- Use constexpr for compile-time computation
- Range-based for loops: for (const auto& item : container)
- Naming: snake_case (Google style) or camelCase (Microsoft style) — be consistent
- Tests: Google Test with TEST(), EXPECT_EQ(), ASSERT_TRUE()
"""

RUBY_SKILLS = """
## Ruby Coding Standards:
- Duck typing: check behavior, not type
- Use blocks, procs, and lambdas for functional patterns
- Prefer symbols over strings for hash keys: { name: "Alice" }
- Use modules for mixins and namespacing
- attr_reader, attr_writer, attr_accessor for properties
- Naming: snake_case for methods/variables, PascalCase for classes, SCREAMING_CASE for constants
- Tests: RSpec with describe/context/it blocks, expect().to matcher
"""

PHP_SKILLS = """
## PHP Coding Standards:
- Use strict types: declare(strict_types=1)
- Type declarations on all parameters and return types (PHP 8+)
- Use enums (PHP 8.1+) instead of string constants
- Readonly properties and constructor promotion
- Naming: camelCase for methods, PascalCase for classes, PSR-12 style
- Tests: PHPUnit with @test annotation or test_ prefix
"""

SWIFT_SKILLS = """
## Swift Coding Standards:
- Use structs for value types, classes only when reference semantics needed
- Guard clauses with guard...else for early returns
- Optional chaining: value?.method() instead of if let
- Protocol-oriented programming over class inheritance
- Use Result<Success, Failure> for error handling
- Naming: camelCase for functions/properties, PascalCase for types/protocols
- Tests: XCTest with func testMethodName(), XCTAssertEqual
"""

SCALA_SKILLS = """
## Scala Coding Standards:
- Use case class for immutable data, sealed trait for algebraic types
- Pattern matching with match/case
- Prefer immutable collections (List, Map, Set)
- Use for-comprehensions for monadic operations
- Implicit conversions sparingly, prefer extension methods (Scala 3)
- Naming: camelCase for methods, PascalCase for types
- Tests: ScalaTest with FunSpec or WordSpec style
"""

ELIXIR_SKILLS = """
## Elixir Coding Standards:
- Use pattern matching in function heads for dispatch
- Pipe operator |> for data transformation chains
- GenServer for stateful processes
- Supervisor trees for fault tolerance
- Use with for happy-path chains with error handling
- Naming: snake_case for functions/variables, PascalCase for modules
- Tests: ExUnit with test "description" do...end blocks
"""

# Map language name to skills string
LANGUAGE_SKILLS: dict[str, str] = {
    "rust": RUST_SKILLS,
    "java": JAVA_SKILLS,
    "kotlin": KOTLIN_SKILLS,
    "csharp": CSHARP_SKILLS,
    "cpp": CPP_SKILLS,
    "ruby": RUBY_SKILLS,
    "php": PHP_SKILLS,
    "swift": SWIFT_SKILLS,
    "scala": SCALA_SKILLS,
    "elixir": ELIXIR_SKILLS,
    # Python, Go, TypeScript have their own dedicated skill files
}


def get_language_skills(lang: str) -> str | None:
    """Return language-specific skills for injection, or None if not available."""
    return LANGUAGE_SKILLS.get(lang)
