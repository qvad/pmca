# Core Software Design Patterns

## 1. Strategy Pattern
**Intent**: Define a family of algorithms, encapsulate each one, and make them interchangeable.
**Use when**: You have multiple ways to perform a task (e.g., different compression algorithms, different auth methods).
**Interface**:
- `Strategy` interface with `execute()` method.
- `Context` class that holds a reference to a `Strategy`.

## 2. Observer Pattern
**Intent**: Define a one-to-many dependency so that when one object changes state, all its dependents are notified.
**Use when**: Event-driven systems, UI updates, state change notifications.
**Interface**:
- `Subject`: `attach(observer)`, `detach(observer)`, `notify()`.
- `Observer`: `update(data)`.

## 3. Factory Method
**Intent**: Define an interface for creating an object, but let subclasses decide which class to instantiate.
**Use when**: Decoupling object creation from its use.

## 4. Repository Pattern
**Intent**: Mediate between the domain and data mapping layers using a collection-like interface.
**Use when**: Database access, API data fetching.
**Interface**:
- `get(id)`, `list(filter)`, `add(entity)`, `delete(id)`.

## 5. Middleware / Chain of Responsibility
**Intent**: Pass requests along a chain of handlers.
**Use when**: Request processing, logging, validation.
