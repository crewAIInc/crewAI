# State Management Documentation

## Overview

The crewAI frontend uses Zustand for state management, providing a simple yet powerful way to handle both traditional crewAI agents and AutoGen agents. The store is designed to be type-safe, performant, and easy to debug.

## Store Structure

```typescript
// types/store.ts
interface Agent {
  id: string;
  name: string;
  role: string;
  type: 'crewai' | 'autogen';
  backstory?: string;
  goal?: string;
  llmConfig?: any;
}

interface Task {
  id: string;
  description: string;
  agent: Agent;
  autogenRecipient?: Agent; // New field for AutoGen tasks
  status: 'pending' | 'in_progress' | 'completed';
  output?: string;
}

interface StoreState {
  agents: Agent[];
  tasks: Task[];
  isLoading: boolean;
  error: string | null;
}
```

## Store Implementation

## Overview

This document details the state management implementation using Zustand for the crewAI frontend. The state management system handles all application state, including data from the API and real-time updates via WebSocket.

## Implementation Details

### Directory Structure
```
lib/
├── store/
│   ├── index.ts         # Main store export
│   ├── types.ts         # Store type definitions
│   ├── crewStore.ts     # Crew-related state
│   ├── taskStore.ts     # Task-related state
│   └── uiStore.ts       # UI-related state
```

### Type Definitions (`types.ts`)

```typescript
// lib/store/types.ts

import { Crew, Task, Agent } from '../api/types';

export interface CrewState {
  crews: Map<number, Crew>;
  loading: boolean;
  error: string | null;
  selectedCrewId: number | null;
}

export interface TaskState {
  tasks: Map<number, Task>;
  loading: boolean;
  error: string | null;
  selectedTaskId: number | null;
}

export interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  notifications: Array<{
    id: string;
    message: string;
    type: 'success' | 'error' | 'info';
  }>;
}

export interface StoreState extends CrewState, TaskState, UIState {
  // Actions
  fetchCrews: () => Promise<void>;
  createCrew: (data: Omit<Crew, 'id'>) => Promise<void>;
  updateCrew: (id: number, data: Partial<Crew>) => Promise<void>;
  deleteCrew: (id: number) => Promise<void>;
  selectCrew: (id: number | null) => void;
  
  fetchTasks: () => Promise<void>;
  createTask: (data: Omit<Task, 'id'>) => Promise<void>;
  updateTask: (id: number, data: Partial<Task>) => Promise<void>;
  deleteTask: (id: number) => Promise<void>;
  selectTask: (id: number | null) => void;
  
  toggleSidebar: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
  addNotification: (message: string, type: 'success' | 'error' | 'info') => void;
  removeNotification: (id: string) => void;
}
```

### Store Implementation

```typescript
// lib/store/index.ts

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { apiClient } from '../api/client';
import { websocketClient } from '../api/websocket';
import { StoreState } from './types';

const useStore = create<StoreState>()(
  devtools(
    (set, get) => ({
      // Initial state
      crews: new Map(),
      tasks: new Map(),
      loading: false,
      error: null,
      selectedCrewId: null,
      selectedTaskId: null,
      sidebarOpen: true,
      theme: 'light',
      notifications: [],

      // Crew actions
      fetchCrews: async () => {
        set({ loading: true, error: null });
        try {
          const crews = await apiClient.getCrews();
          set({
            crews: new Map(crews.map(crew => [crew.id, crew])),
            loading: false
          });
        } catch (error: any) {
          set({
            error: error.message,
            loading: false
          });
          get().addNotification(error.message, 'error');
        }
      },

      createCrew: async (data) => {
        set({ loading: true, error: null });
        try {
          const crew = await apiClient.createCrew(data);
          set(state => ({
            crews: new Map(state.crews).set(crew.id, crew),
            loading: false
          }));
          get().addNotification('Crew created successfully', 'success');
        } catch (error: any) {
          set({
            error: error.message,
            loading: false
          });
          get().addNotification(error.message, 'error');
        }
      },

      updateCrew: async (id, data) => {
        set({ loading: true, error: null });
        try {
          const crew = await apiClient.updateCrew(id, data);
          set(state => ({
            crews: new Map(state.crews).set(crew.id, crew),
            loading: false
          }));
          get().addNotification('Crew updated successfully', 'success');
        } catch (error: any) {
          set({
            error: error.message,
            loading: false
          });
          get().addNotification(error.message, 'error');
        }
      },

      deleteCrew: async (id) => {
        set({ loading: true, error: null });
        try {
          await apiClient.deleteCrew(id);
          set(state => {
            const crews = new Map(state.crews);
            crews.delete(id);
            return { crews, loading: false };
          });
          get().addNotification('Crew deleted successfully', 'success');
        } catch (error: any) {
          set({
            error: error.message,
            loading: false
          });
          get().addNotification(error.message, 'error');
        }
      },

      selectCrew: (id) => {
        set({ selectedCrewId: id });
      },

      // Task actions
      fetchTasks: async () => {
        set({ loading: true, error: null });
        try {
          const tasks = await apiClient.getTasks();
          set({
            tasks: new Map(tasks.map(task => [task.id, task])),
            loading: false
          });
        } catch (error: any) {
          set({
            error: error.message,
            loading: false
          });
          get().addNotification(error.message, 'error');
        }
      },

      // UI actions
      toggleSidebar: () => {
        set(state => ({ sidebarOpen: !state.sidebarOpen }));
      },

      setTheme: (theme) => {
        set({ theme });
      },

      addNotification: (message, type) => {
        const id = Date.now().toString();
        set(state => ({
          notifications: [...state.notifications, { id, message, type }]
        }));
        // Auto-remove notification after 5 seconds
        setTimeout(() => {
          get().removeNotification(id);
        }, 5000);
      },

      removeNotification: (id) => {
        set(state => ({
          notifications: state.notifications.filter(n => n.id !== id)
        }));
      }
    })
  )
);

// Set up WebSocket listeners
websocketClient.connect({
  crew_created: (crew) => {
    useStore.setState(state => ({
      crews: new Map(state.crews).set(crew.id, crew)
    }));
  },
  crew_updated: (crew) => {
    useStore.setState(state => ({
      crews: new Map(state.crews).set(crew.id, crew)
    }));
  },
  crew_deleted: (crewId) => {
    useStore.setState(state => {
      const crews = new Map(state.crews);
      crews.delete(crewId);
      return { crews };
    });
  },
  // ... similar handlers for tasks
});

export default useStore;
```

## Usage Examples

### Basic Usage in Components

```typescript
// components/CrewList.tsx
'use client';

import { useEffect } from 'react';
import useStore from '@/lib/store';

export default function CrewList() {
  const { crews, loading, error, fetchCrews } = useStore();

  useEffect(() => {
    fetchCrews();
  }, [fetchCrews]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {Array.from(crews.values()).map(crew => (
        <div key={crew.id}>
          <h3>{crew.name}</h3>
          <p>Status: {crew.status}</p>
        </div>
      ))}
    </div>
  );
}
```

### Using Actions

```typescript
// components/CreateCrewButton.tsx
'use client';

import useStore from '@/lib/store';

export default function CreateCrewButton() {
  const createCrew = useStore(state => state.createCrew);

  const handleClick = async () => {
    await createCrew({
      name: 'New Crew',
      agents: [],
      status: 'Idle'
    });
  };

  return (
    <button onClick={handleClick}>
      Create New Crew
    </button>
  );
}
```

## Best Practices

1. **Selective Subscription**: Only subscribe to the state you need in each component
2. **Action Encapsulation**: Keep all state modifications within store actions
3. **Error Handling**: Use the notification system for user feedback
4. **Real-time Updates**: Handle WebSocket events through store actions
5. **Type Safety**: Maintain strict TypeScript types for all state and actions

## Performance Considerations

1. **State Structure**: Use Maps for O(1) lookups of crews and tasks
2. **Selective Updates**: Only update affected parts of the state
3. **Memoization**: Use React.memo and useMemo when necessary
4. **Batched Updates**: Combine multiple state updates when possible
