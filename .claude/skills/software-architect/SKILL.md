---
name: software-architect
description: |
  Guide for writing clean, maintainable code following SOLID principles.
  Use this skill when: (1) Writing a function that does multiple things (validate, save, notify),
  (2) Adding if/elif chains for new feature variations, (3) Creating classes with inheritance,
  (4) A class/function requires dependencies it doesn't fully use, (5) Code is hard to test
  because it creates its own dependencies, (6) Refactoring for better structure.
  Apply these principles to ensure code is easy to understand, modify, and test.
---

# Clean Code with SOLID Principles

**Core Philosophy: Write code that is easy to understand, change, and test.**

SOLID is an acronym for five principles that help you write better code. Apply these every time you write functions or classes.

## Quick Checklist Before Writing Code

| Question to Ask | If Yes, You're Good | If No, Refactor |
|-----------------|---------------------|-----------------|
| Does this function/class do ONE thing? | ✓ | Split it up |
| Can I add features without changing existing code? | ✓ | Use abstractions |
| Can I replace this with a similar component? | ✓ | Fix the contract |
| Am I only using what I need? | ✓ | Create smaller interfaces |
| Do I depend on abstractions, not specifics? | ✓ | Inject dependencies |

---

## S - Single Responsibility Principle

**"A function or class should do one thing and do it well."**

### Why It Matters
- Easier to understand (one purpose = one mental model)
- Easier to test (test one thing at a time)
- Easier to change (change one thing without breaking others)

### Bad Example

```python
def process_user_registration(email: str, password: str) -> dict:
    # Validates email
    if "@" not in email:
        raise ValueError("Invalid email")

    # Validates password
    if len(password) < 8:
        raise ValueError("Password too short")

    # Creates user in database
    user_id = database.insert("users", {"email": email, "password": hash(password)})

    # Sends welcome email
    smtp.send(email, "Welcome!", "Thanks for joining!")

    # Logs the registration
    logger.info(f"New user registered: {email}")

    return {"user_id": user_id, "email": email}
```

**Problem**: This function does 5 different things. If you need to change how emails are sent, you're touching the same code that handles validation and database operations.

### Good Example

```python
def validate_email(email: str) -> bool:
    """Check if email format is valid."""
    return "@" in email and "." in email

def validate_password(password: str) -> bool:
    """Check if password meets requirements."""
    return len(password) >= 8

def create_user(email: str, password: str) -> str:
    """Create user in database and return user ID."""
    return database.insert("users", {"email": email, "password": hash(password)})

def send_welcome_email(email: str) -> None:
    """Send welcome email to new user."""
    smtp.send(email, "Welcome!", "Thanks for joining!")

def register_user(email: str, password: str) -> dict:
    """Orchestrate the user registration process."""
    if not validate_email(email):
        raise ValueError("Invalid email")
    if not validate_password(password):
        raise ValueError("Password too short")

    user_id = create_user(email, password)
    send_welcome_email(email)

    return {"user_id": user_id, "email": email}
```

**Benefits**:
- Each function is easy to understand
- You can test `validate_email` without a database
- You can change email sending without touching validation

---

## O - Open/Closed Principle

**"Code should be open for extension but closed for modification."**

### Why It Matters
- Add new features without changing existing code
- Reduces risk of breaking things that already work
- Makes your code more flexible

### Bad Example

```python
def calculate_discount(customer_type: str, amount: float) -> float:
    """Calculate discount based on customer type."""
    if customer_type == "regular":
        return amount * 0.05
    elif customer_type == "premium":
        return amount * 0.10
    elif customer_type == "vip":
        return amount * 0.20
    else:
        return 0.0

# Problem: To add a new customer type, you MUST modify this function
# What if you forget a case? What if this function is used everywhere?
```

### Good Example

```python
# Define discount strategies
DISCOUNT_RATES = {
    "regular": 0.05,
    "premium": 0.10,
    "vip": 0.20,
}

def calculate_discount(customer_type: str, amount: float) -> float:
    """Calculate discount based on customer type."""
    rate = DISCOUNT_RATES.get(customer_type, 0.0)
    return amount * rate

# To add a new customer type, just add to the dictionary:
# DISCOUNT_RATES["enterprise"] = 0.25
# No need to modify the function!
```

---

## L - Liskov Substitution Principle

**"If you replace a parent with a child, things should still work."**

### Why It Matters
- Ensures your code is truly reusable
- Prevents unexpected bugs when using inheritance
- Makes your class hierarchies trustworthy

### Bad Example

```python
class Bird:
    def fly(self) -> str:
        return "Flying high!"

class Penguin(Bird):
    def fly(self) -> str:
        raise Exception("Penguins can't fly!")  # BREAKS the contract!

def make_bird_fly(bird: Bird) -> str:
    return bird.fly()

# This will crash unexpectedly:
penguin = Penguin()
make_bird_fly(penguin)  # Exception: Penguins can't fly!
```

**Problem**: `Penguin` inherits from `Bird` but can't fulfill the `fly()` contract. Code expecting a `Bird` will break.

### Good Example

```python
class Bird:
    def move(self) -> str:
        return "Moving"

class FlyingBird(Bird):
    def fly(self) -> str:
        return "Flying high!"

class SwimmingBird(Bird):
    def swim(self) -> str:
        return "Swimming!"

class Eagle(FlyingBird):
    def fly(self) -> str:
        return "Soaring through the sky!"

class Penguin(SwimmingBird):
    def swim(self) -> str:
        return "Swimming gracefully!"

# Now each bird type can be used correctly:
def make_bird_fly(bird: FlyingBird) -> str:
    return bird.fly()

def make_bird_swim(bird: SwimmingBird) -> str:
    return bird.swim()

eagle = Eagle()
make_bird_fly(eagle)  # Works!

penguin = Penguin()
make_bird_swim(penguin)  # Works!
```

### Simple Rule
If your child class needs to throw an exception or return `None` for a method that the parent defines, you probably have the wrong inheritance structure.

---

## I - Interface Segregation Principle

**"Don't force code to depend on things it doesn't use."**

### Why It Matters
- Keeps your code focused and lean
- Reduces unnecessary dependencies
- Makes testing easier

### Bad Example

```python
class Worker:
    def work(self) -> str:
        pass

    def eat(self) -> str:
        pass

    def sleep(self) -> str:
        pass

class Robot(Worker):
    def work(self) -> str:
        return "Working..."

    def eat(self) -> str:
        raise Exception("Robots don't eat!")  # Forced to implement this!

    def sleep(self) -> str:
        raise Exception("Robots don't sleep!")  # Forced to implement this!
```

**Problem**: `Robot` is forced to implement `eat()` and `sleep()` even though it doesn't need them.

### Good Example

```python
class Workable:
    def work(self) -> str:
        pass

class Eatable:
    def eat(self) -> str:
        pass

class Sleepable:
    def sleep(self) -> str:
        pass

class Human(Workable, Eatable, Sleepable):
    def work(self) -> str:
        return "Working..."

    def eat(self) -> str:
        return "Eating lunch..."

    def sleep(self) -> str:
        return "Sleeping..."

class Robot(Workable):  # Only implements what it needs!
    def work(self) -> str:
        return "Working 24/7..."
```

### Practical Application: Function Parameters

```python
# Bad: Function takes more than it needs
def send_notification(user: User) -> None:
    # Only uses user.email, but requires entire User object
    email_service.send(user.email, "Hello!")

# Good: Function takes only what it needs
def send_notification(email: str) -> None:
    email_service.send(email, "Hello!")

# Now you can call it without having a full User object:
send_notification("user@example.com")
```

---

## D - Dependency Inversion Principle

**"Depend on abstractions, not concrete implementations."**

### Why It Matters
- Makes code flexible and swappable
- Makes testing much easier (use fakes/mocks)
- Reduces coupling between components

### Bad Example

```python
class EmailService:
    def send(self, to: str, message: str) -> None:
        # Sends email via SMTP
        smtp_server.send(to, message)

class UserRegistration:
    def __init__(self):
        self.email_service = EmailService()  # HARD-CODED dependency!

    def register(self, email: str, password: str) -> None:
        # Create user...
        user_id = create_user(email, password)
        # Send welcome email
        self.email_service.send(email, "Welcome!")

# Problem: Can't test without actually sending emails!
# Problem: Can't switch to a different email provider easily
```

### Good Example

```python
from abc import ABC, abstractmethod

# 1. Define what you need (abstraction)
class NotificationService(ABC):
    @abstractmethod
    def send(self, to: str, message: str) -> None:
        pass

# 2. Create implementations
class EmailNotificationService(NotificationService):
    def send(self, to: str, message: str) -> None:
        smtp_server.send(to, message)

class SMSNotificationService(NotificationService):
    def send(self, to: str, message: str) -> None:
        sms_gateway.send(to, message)

# 3. Depend on the abstraction, not the implementation
class UserRegistration:
    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service  # INJECTED!

    def register(self, email: str, password: str) -> None:
        user_id = create_user(email, password)
        self.notification_service.send(email, "Welcome!")

# Usage - you choose which implementation to use:
email_service = EmailNotificationService()
registration = UserRegistration(email_service)

# For testing - use a fake:
class FakeNotificationService(NotificationService):
    def __init__(self):
        self.sent_messages = []

    def send(self, to: str, message: str) -> None:
        self.sent_messages.append((to, message))

# Test without sending real emails:
fake_service = FakeNotificationService()
registration = UserRegistration(fake_service)
registration.register("test@example.com", "password123")
assert len(fake_service.sent_messages) == 1
```

---

## Quick Function Guidelines

| Guideline | Recommendation |
|-----------|----------------|
| **Length** | Keep functions under 20 lines |
| **Arguments** | Prefer 0-3 arguments |
| **Naming** | Use verb + noun: `calculate_total`, `send_email` |
| **One thing** | Each function does exactly one thing |
| **No surprises** | Function does what its name says, nothing more |

---

## Summary: SOLID at a Glance

| Principle | In Simple Terms | Quick Test |
|-----------|----------------|------------|
| **S**ingle Responsibility | One function = one job | Can you describe it without saying "and"? |
| **O**pen/Closed | Add features, don't modify | Can you extend without editing? |
| **L**iskov Substitution | Children honor parent's promises | Does every child work where parent works? |
| **I**nterface Segregation | Don't force unused dependencies | Is everything you require actually used? |
| **D**ependency Inversion | Depend on abstractions | Can you swap implementations easily? |

---

## When to Apply

- **Always apply S**: Every function should do one thing
- **Apply O when**: You find yourself adding `if/elif` chains for new cases
- **Apply L when**: Using inheritance
- **Apply I when**: Your interfaces have methods some implementers don't need
- **Apply D when**: You want testable code or need flexibility
