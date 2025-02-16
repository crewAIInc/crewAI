# CrewList Component

The `CrewList` component is a core component of the crewAI frontend that displays a responsive grid of crews with real-time updates. It includes comprehensive accessibility features and handles various states (loading, error, empty) gracefully.

## Features

- Responsive grid layout (1 column on mobile, 2 on tablet, 3 on desktop)
- Real-time updates through WebSocket integration
- Loading state with skeleton animation
- Error state with user-friendly message
- Empty state with call-to-action
- Accessible with proper ARIA labels and roles
- Keyboard navigation support
- Mobile-first design

## Usage

```tsx
import { CrewList } from '@/components/CrewList';

// Basic usage
const MyPage = () => {
    return <CrewList />;
};
```

## Component Structure

The `CrewList` component is composed of several sub-components for better organization and maintainability:

### StatusBadge

Displays the crew's status with appropriate color coding:
- Active: Green
- Running: Blue
- Completed: Gray
- Failed: Red

### CrewCard

Displays individual crew information including:
- Name
- Status
- Description (truncated to 2 lines)
- Agent count
- Task count
- Creation date

### LoadingState

Shows a skeleton loading animation with placeholder cards.

### ErrorState

Displays error messages in a user-friendly format.

### EmptyState

Shows when no crews are available, with a button to create a new crew.

## Props

The `CrewList` component doesn't accept any props as it manages its own state through the Zustand store.

## State Management

The component uses the following state from the Zustand store:
- `crews`: Array of crew objects
- `isLoading`: Boolean loading state
- `error`: Error object if any
- `fetchCrews`: Function to fetch crews

## Accessibility

The component implements the following accessibility features:

- Proper heading hierarchy
- ARIA labels for interactive elements
- Role attributes for semantic HTML
- Status messages for loading and error states
- Keyboard navigation support
- Color contrast compliance
- Screen reader support

## Example

```tsx
// Page component example
import { CrewList } from '@/components/CrewList';

const CrewsPage = () => {
    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-2xl font-bold mb-6">Your Crews</h1>
            <CrewList />
        </div>
    );
};

export default CrewsPage;
```

## Dependencies

- Next.js
- React
- Zustand (for state management)
- Tailwind CSS (for styling)

## Best Practices

1. **Performance**
   - Uses CSS Grid for responsive layout
   - Implements skeleton loading for better UX
   - Optimizes re-renders with proper dependency arrays

2. **Accessibility**
   - Follows WCAG 2.1 guidelines
   - Implements proper ARIA attributes
   - Ensures keyboard navigation
   - Provides clear feedback for all states

3. **Maintainability**
   - Separates concerns into sub-components
   - Uses TypeScript for type safety
   - Follows consistent naming conventions
   - Implements proper error boundaries

4. **Responsiveness**
   - Mobile-first approach
   - Responsive grid layout
   - Adaptive typography
   - Touch-friendly targets

## Related Components

- `CrewDetail`: Displays detailed information about a specific crew
- `CreateCrewForm`: Form for creating new crews
- `TaskList`: Displays tasks associated with a crew 