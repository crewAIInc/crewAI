# Form Handling Documentation

## Overview

This document details the implementation of forms in the crewAI frontend using `react-hook-form`, with a focus on accessibility, type safety, and proper error handling. All components follow the DRY principle and implement proper accessibility features.

## Implementation Details

### Directory Structure
```
components/
├── forms/
│   ├── CrewForm.tsx
│   ├── TaskForm.tsx
│   └── common/
│       ├── FormField.tsx
│       ├── FormSelect.tsx
│       └── FormError.tsx
```

### Common Form Components

```typescript
// components/forms/common/FormField.tsx
import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

interface FormFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  description?: string;
}

export const FormField = forwardRef<HTMLInputElement, FormFieldProps>(
  ({ label, error, description, className, id, ...props }, ref) => {
    const inputId = id || props.name;
    const descriptionId = `${inputId}-description`;
    const errorId = `${inputId}-error`;

    return (
      <div className="space-y-2">
        <label
          htmlFor={inputId}
          className="block text-sm font-medium text-gray-700 dark:text-gray-200"
        >
          {label}
        </label>
        <input
          ref={ref}
          id={inputId}
          className={cn(
            "w-full rounded-md border border-gray-300 px-4 py-2 text-sm",
            "focus:border-primary-500 focus:ring-2 focus:ring-primary-500",
            "disabled:cursor-not-allowed disabled:bg-gray-100",
            error && "border-red-500 focus:border-red-500 focus:ring-red-500",
            className
          )}
          aria-describedby={`${description ? descriptionId : ''} ${
            error ? errorId : ''
          }`.trim()}
          aria-invalid={!!error}
          {...props}
        />
        {description && (
          <p
            id={descriptionId}
            className="text-sm text-gray-500 dark:text-gray-400"
          >
            {description}
          </p>
        )}
        {error && (
          <p
            id={errorId}
            className="text-sm text-red-500"
            role="alert"
          >
            {error}
          </p>
        )}
      </div>
    );
  }
);

FormField.displayName = 'FormField';
```

### Task Form Implementation with AutoGen Support

```typescript
// components/forms/TaskForm.tsx
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useStore } from '@/lib/store';
import type { Task, Agent } from '@/lib/types';

const taskSchema = z.object({
  description: z.string()
    .min(10, 'Description must be at least 10 characters')
    .max(500, 'Description must be less than 500 characters'),
  agent: z.string(),
  autogenRecipient: z.string().optional(),
  status: z.enum(['pending', 'in_progress', 'completed'])
});

type TaskFormData = z.infer<typeof taskSchema>;

interface TaskFormProps {
  initialData?: Partial<Task>;
  onSuccess?: () => void;
  onCancel?: () => void;
}

export const TaskForm = ({ initialData, onSuccess, onCancel }: TaskFormProps) => {
  const { agents, createTask, updateTask } = useStore();
  
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
    reset
  } = useForm<TaskFormData>({
    resolver: zodResolver(taskSchema),
    defaultValues: {
      description: initialData?.description || '',
      agent: initialData?.agent?.id || '',
      autogenRecipient: initialData?.autogenRecipient?.id || '',
      status: initialData?.status || 'pending'
    }
  });

  const selectedAgentId = watch('agent');
  const selectedAgent = agents.find(a => a.id === selectedAgentId);
  const isAutogenAgent = selectedAgent?.type === 'autogen';

  // Filter available recipients (only AutoGen agents)
  const availableRecipients = agents.filter(a => 
    a.type === 'autogen' && a.id !== selectedAgentId
  );

  const handleFormSubmit = async (data: TaskFormData) => {
    try {
      if (initialData?.id) {
        await updateTask(initialData.id, data);
      } else {
        await createTask(data);
      }
      reset();
      onSuccess?.();
    } catch (error) {
      // Error handling managed by store
    }
  };

  return (
    <form 
      onSubmit={handleSubmit(handleFormSubmit)}
      className="space-y-6"
      noValidate
    >
      <div className="space-y-4">
        <div>
          <label
            htmlFor="description"
            className="block text-sm font-medium"
          >
            Task Description
          </label>
          <textarea
            id="description"
            {...register('description')}
            className={cn(
              "mt-1 block w-full rounded-md border-gray-300",
              errors.description && "border-red-500"
            )}
            rows={4}
            aria-invalid={!!errors.description}
            aria-describedby={errors.description ? "description-error" : undefined}
          />
          {errors.description && (
            <p
              id="description-error"
              className="mt-1 text-sm text-red-500"
              role="alert"
            >
              {errors.description.message}
            </p>
          )}
        </div>

        <div>
          <label
            htmlFor="agent"
            className="block text-sm font-medium"
          >
            Assign Agent
          </label>
          <select
            id="agent"
            {...register('agent')}
            className="mt-1 block w-full rounded-md"
            aria-invalid={!!errors.agent}
          >
            <option value="">Select an agent</option>
            {agents.map(agent => (
              <option key={agent.id} value={agent.id}>
                {agent.name} ({agent.type})
              </option>
            ))}
          </select>
          {errors.agent && (
            <p className="mt-1 text-sm text-red-500" role="alert">
              {errors.agent.message}
            </p>
          )}
        </div>

        {isAutogenAgent && (
          <div>
            <label
              htmlFor="autogenRecipient"
              className="block text-sm font-medium"
            >
              AutoGen Recipient (Optional)
            </label>
            <select
              id="autogenRecipient"
              {...register('autogenRecipient')}
              className="mt-1 block w-full rounded-md"
              aria-invalid={!!errors.autogenRecipient}
            >
              <option value="">None</option>
              {availableRecipients.map(agent => (
                <option key={agent.id} value={agent.id}>
                  {agent.name}
                </option>
              ))}
            </select>
            {errors.autogenRecipient && (
              <p className="mt-1 text-sm text-red-500" role="alert">
                {errors.autogenRecipient.message}
              </p>
            )}
          </div>
        )}

        <div>
          <label
            htmlFor="status"
            className="block text-sm font-medium"
          >
            Status
          </label>
          <select
            id="status"
            {...register('status')}
            className="mt-1 block w-full rounded-md"
          >
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
          </select>
        </div>
      </div>

      <div className="flex justify-end space-x-4">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
        )}
        <button
          type="submit"
          disabled={isSubmitting}
          className={cn(
            "px-4 py-2 bg-blue-500 text-white rounded-md",
            "hover:bg-blue-600 focus:outline-none focus:ring-2",
            "focus:ring-blue-500 focus:ring-offset-2",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isSubmitting ? 'Saving...' : initialData ? 'Update Task' : 'Create Task'}
        </button>
      </div>
    </form>
  );
};
```

### Crew Form Implementation

```typescript
// components/forms/CrewForm.tsx
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { FormField } from './common/FormField';
import useStore from '@/lib/store';
import { Crew } from '@/lib/api/types';

// Validation schema
const crewSchema = z.object({
  name: z.string()
    .min(3, 'Name must be at least 3 characters')
    .max(50, 'Name must be less than 50 characters'),
  agents: z.string()
    .transform((val) => val.split(',').map(Number).filter(n => !isNaN(n))),
  status: z.enum(['Active', 'Idle'])
});

type CrewFormData = z.infer<typeof crewSchema>;

interface CrewFormProps {
  initialData?: Partial<Crew>;
  onSuccess?: () => void;
}

export const CrewForm = ({ initialData, onSuccess }: CrewFormProps) => {
  const { createCrew, updateCrew } = useStore();
  
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset
  } = useForm<CrewFormData>({
    resolver: zodResolver(crewSchema),
    defaultValues: {
      name: initialData?.name || '',
      agents: initialData?.agents?.join(',') || '',
      status: initialData?.status || 'Idle'
    }
  });

  const handleFormSubmit = async (data: CrewFormData) => {
    try {
      if (initialData?.id) {
        await updateCrew(initialData.id, {
          ...data,
          agents: data.agents // Already transformed by zod
        });
      } else {
        await createCrew({
          ...data,
          agents: data.agents // Already transformed by zod
        });
      }
      reset();
      onSuccess?.();
    } catch (error) {
      // Error handling is managed by the store
    }
  };

  return (
    <form 
      onSubmit={handleSubmit(handleFormSubmit)}
      className="space-y-6"
      noValidate
    >
      <FormField
        label="Crew Name"
        {...register('name')}
        error={errors.name?.message}
        aria-required="true"
        data-testid="crew-name-input"
      />

      <FormField
        label="Agent IDs"
        {...register('agents')}
        error={errors.agents?.message}
        description="Enter agent IDs separated by commas (e.g., 1,2,3)"
        data-testid="crew-agents-input"
      />

      <div className="space-y-2">
        <label
          htmlFor="status"
          className="block text-sm font-medium text-gray-700 dark:text-gray-200"
        >
          Status
        </label>
        <select
          id="status"
          {...register('status')}
          className={cn(
            "w-full rounded-md border border-gray-300 px-4 py-2 text-sm",
            "focus:border-primary-500 focus:ring-2 focus:ring-primary-500",
            errors.status && "border-red-500"
          )}
          aria-invalid={!!errors.status}
          data-testid="crew-status-select"
        >
          <option value="Active">Active</option>
          <option value="Idle">Idle</option>
        </select>
        {errors.status && (
          <p className="text-sm text-red-500" role="alert">
            {errors.status.message}
          </p>
        )}
      </div>

      <button
        type="submit"
        disabled={isSubmitting}
        className={cn(
          "w-full rounded-md bg-primary-500 px-4 py-2 text-white",
          "hover:bg-primary-600 focus:outline-none focus:ring-2",
          "focus:ring-primary-500 focus:ring-offset-2",
          "disabled:cursor-not-allowed disabled:opacity-50"
        )}
        data-testid="crew-submit-button"
      >
        {isSubmitting ? 'Saving...' : initialData ? 'Update Crew' : 'Create Crew'}
      </button>
    </form>
  );
};
```

### Usage Example

```typescript
// app/crews/new/page.tsx
'use client';

import { CrewForm } from '@/components/forms/CrewForm';
import { useRouter } from 'next/navigation';

export default function NewCrewPage() {
  const router = useRouter();

  const handleSuccess = () => {
    router.push('/crews');
  };

  return (
    <div className="container mx-auto max-w-2xl py-8">
      <h1 className="mb-8 text-2xl font-bold">Create New Crew</h1>
      <CrewForm onSuccess={handleSuccess} />
    </div>
  );
}
```

## Accessibility Features

1. **ARIA Attributes**
   - `aria-describedby` for field descriptions and errors
   - `aria-invalid` for invalid fields
   - `aria-required` for required fields
   - `role="alert"` for error messages

2. **Keyboard Navigation**
   - All interactive elements are focusable
   - Logical tab order
   - Focus management after form submission

3. **Error Handling**
   - Clear error messages
   - Visual indicators for invalid fields
   - Screen reader announcements for errors

## Form Validation

1. **Client-side Validation**
   - Zod schema validation
   - Real-time field validation
   - Custom validation rules

2. **Server-side Validation**
   - API response handling
   - Error message display
   - Field-level error mapping

## Testing

```typescript
// __tests__/components/CrewForm.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CrewForm } from '@/components/forms/CrewForm';

describe('CrewForm', () => {
  it('validates required fields', async () => {
    render(<CrewForm />);
    
    fireEvent.click(screen.getByTestId('crew-submit-button'));
    
    await waitFor(() => {
      expect(screen.getByText(/Name must be at least 3 characters/i)).toBeInTheDocument();
    });
  });

  it('submits form with valid data', async () => {
    const onSuccess = jest.fn();
    render(<CrewForm onSuccess={onSuccess} />);
    
    fireEvent.change(screen.getByTestId('crew-name-input'), {
      target: { value: 'Test Crew' }
    });
    
    fireEvent.change(screen.getByTestId('crew-agents-input'), {
      target: { value: '1,2,3' }
    });
    
    fireEvent.click(screen.getByTestId('crew-submit-button'));
    
    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalled();
    });
  });
});
```

## Best Practices

1. **Component Organization**
   - Reusable form components
   - Consistent styling
   - Clear component hierarchy

2. **Type Safety**
   - TypeScript interfaces
   - Zod schema validation
   - Proper type inference

3. **Error Handling**
   - Comprehensive error states
   - User-friendly error messages
   - Proper error propagation

4. **Performance**
   - Controlled vs Uncontrolled components
   - Form state optimization
   - Proper cleanup
