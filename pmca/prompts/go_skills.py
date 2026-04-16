"""Golang Systems Engineering Skills.
Focus: Idiomatic Go, concurrency, and robust error management.
"""

GO_SKILLS = """
## GO OPERATIONAL MANDATES:
1. IDIOMATIC GO: Follow `gofmt` and `golint` conventions. Use `PascalCase` for exports, `camelCase` for internal names.
2. EXPLICIT ERROR HANDLING: ALWAYS check `if err != nil` immediately after any operation that returns an error.
3. INTERFACE-ORIENTED: Define small, focused interfaces (e.g., `Reader`, `Writer`, `Closer`). Prefer "accepting interfaces, returning structs."
4. CONCURRENCY PRIMITIVES: Use Goroutines for concurrent operations. Coordinate via channels or `sync.WaitGroup/Mutex`.
5. ZERO-VALUE AWARENESS: Understand and use zero-values. Avoid uninitialized pointers; use `make()` or `&Struct{}`.
6. SLICE/MAP EFFICIENCY: Pre-allocate with `make([]T, 0, capacity)` for performance.
7. COMPACT DOCS: Use `// FunctionName description` format. Keep it concise.
8. TEST INTEGRITY: Use `testing` package with `t.Errorf/t.Fatalf`. Compute expected states manually.
"""
