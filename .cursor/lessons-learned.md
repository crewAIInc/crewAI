# crewAI Lessons Learned

_This file captures development insights, solutions, and best practices. Each lesson must follow this format:_

## Entry Format

[Timestamp] Category: Issue → Solution → Impact

## Categories

1. Component Development
2. TypeScript Implementation
3. Error Resolution
4. Performance Optimization
5. Security Practices
6. Accessibility Standards
7. Code Organization
8. Testing Strategies

## Priority Levels

- Critical: Security, data integrity, breaking changes
- Important: Accessibility, code organization, testing
- Enhancement: Style, refactoring, developer experience

## Required Information

- Problem description
- Solution details
- Prevention strategy
- Impact assessment
- Code examples (when applicable)
- Related references

### Lessonsusable lessons that improve code quality, prevent common errors, and establish best practices. Cross-reference with @memories.md for context.\*

# Lessons Learned

_Note: This file is updated only upon user request and focuses on capturing important, reusable lessons learned during development. Each entry includes a timestamp, category, and comprehensive explanation to prevent similar issues in the future._

[2024-03-19 10:00] Project Planning: Issue: Complex frontend development requires careful task prioritization and dependency management → Solution: Implemented Frontend Windsurf Wave Rule with clear task hierarchy and dependency tracking in scratchpad → Why: Essential for maintaining focus on highest-impact tasks while ensuring systematic development progress and preventing blocking issues. #planning #methodology

[2024-03-19 10:30] TypeScript Implementation: Issue: Generic WebSocket event handling with type safety in TypeScript → Solution: Implemented type-safe event handlers using TypeScript generics and type assertions, with proper error handling and reconnection logic → Why: Essential for maintaining type safety while allowing flexible event handling and ensuring robust real-time communication. #typescript #websocket #error-handling

[2024-03-19 11:00] State Management: Issue: Complex state management with real-time updates and error handling in TypeScript → Solution: Implemented Zustand store with middleware, type-safe actions, and WebSocket integration, using TypeScript generics for type safety → Why: Essential for maintaining a single source of truth, handling real-time updates efficiently, and providing a great developer experience with proper TypeScript support. #state-management #typescript #websocket

[2024-03-19 11:30] Component Development: Issue: Building accessible, responsive components with proper state handling → Solution: Implemented component composition pattern with separate state components (Loading, Error, Empty) and comprehensive accessibility attributes → Why: Essential for creating maintainable, accessible components that provide a great user experience across all devices and user capabilities. #component #accessibility #ux

[2024-03-19 12:00] Documentation: Issue: Need for comprehensive, standardized component documentation that serves both developers and maintainers → Solution: Implemented MDC (Markdown Content) documentation format with clear sections for purpose, props, state management, usage examples, and best practices → Why: Essential for maintaining a scalable codebase where components are well-documented, easily understood, and consistently implemented across the team. #documentation #best-practices #maintainability

[2024-02-08 17:30] Component Architecture: Issue: Handling complex nested data structures (AutoGen config) in UI components → Fix: Implemented conditional rendering with type checking for optional fields, used semantic HTML with ARIA attributes for better accessibility → Why: Essential for creating robust components that handle varying data shapes while maintaining accessibility standards and type safety. Example: CrewDetail component's agent card implementation with optional AutoGen config display.

[2024-02-08 18:00] State Management: Issue: Handling WebSocket events efficiently in React components → Solution: Leveraged Zustand store's centralized WebSocket event handling instead of component-level subscriptions, ensuring consistent state updates across the application → Why: Critical for maintaining a single source of truth and preventing race conditions in real-time updates. Example: CrewDetail component uses store's WebSocket event handlers to automatically update when crew data changes, eliminating the need for component-level WebSocket subscriptions.

[2024-03-14 10:30] Project Management: Issue: Project renaming requires systematic approach to update all references → Solution: Created comprehensive checklist covering documentation, code, configuration files; used search tools to find all instances → Why: Ensures consistency across codebase and prevents confusion from mixed branding. #project #management #documentation

[2024-03-14 10:30] Form API Integration: Issue: Form submission needed proper error handling, state management, and user feedback → Fix: Implemented comprehensive form submission flow with Zustand store integration, success/error messages, form reset, and redirection → Why: Essential for providing clear feedback to users, maintaining data consistency, and ensuring a smooth user experience in form submissions. #form #api #ux

[2024-03-14 11:00] Testing Strategy: Issue: Complex form testing requires comprehensive coverage of async operations, state management, and user interactions → Solution: Implemented structured test suite with proper mocking of external dependencies (store, router), clear test organization, and thorough coverage of success/error paths → Why: Essential for maintaining form reliability, catching regressions early, and ensuring proper handling of all user scenarios. Example: CreateCrewForm test suite implementation with API integration, state management, and user feedback testing. #testing #forms #best-practices

[2024-03-14 11:30] Component Documentation: Issue: Need for standardized, comprehensive component documentation that covers all aspects of component functionality and implementation → Solution: Implemented MDC-based documentation structure with clear sections for purpose, state management, accessibility, testing, and security, including practical examples and changelogs → Why: Essential for maintaining component knowledge, ensuring consistent implementation, and facilitating team collaboration. Example: CreateCrewForm.mdc documentation implementation with complete coverage of component features and best practices. #documentation #best-practices #maintainability

[2024-03-14 10:45] Testing: Issue: Standard HTML element selectors not working with Shadcn UI components → Fix: Used role-based selectors (getByRole, getAllByRole) with proper ARIA labels and roles for reliable component targeting → Why: Shadcn UI components use complex DOM structures, making direct element selectors unreliable. Role-based selectors are more robust and align with accessibility best practices.
