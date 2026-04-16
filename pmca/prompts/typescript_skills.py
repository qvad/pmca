"""TypeScript & Modern JavaScript Engineering Skills.
Focus: Type safety, async concurrency, and ES6+ idiomatics.
"""

TYPESCRIPT_SKILLS = """
## TYPESCRIPT OPERATIONAL MANDATES:
1. STRICT TYPING: Use explicit interfaces/types for all parameters, return types, and class members. Avoid `any` at all costs; use `unknown` with narrowing if type is truly dynamic.
2. IMPORT INTEGRITY (CRITICAL): Always use top-level `import` statements. NEVER use `require()` inside method bodies (e.g., `const crypto = require('crypto')` is FORBIDDEN).
3. IMMUTABILITY & MUTATION: Strings are primitives. If you "modify" a string, you must return the new value or assign it to an object property. DO NOT reassign a local string parameter and expect it to persist (No-op bug).
4. ASYNC AWARENESS: Use `async/await` exclusively for asynchronous operations. Handle `Promise.all` for concurrent operations.
5. MODERN IDIOMATICS: Use destructuring, template literals, and arrow functions where appropriate.
6. EXPORT HYGIENE: Use named exports (`export class X`) instead of default exports.
7. TEST INTEGRITY: Use `jest` style assertions. Manually trace state before writing `expect`.
"""
