"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import * as api from "@/lib/apiClient";
import { ApiError } from "@/lib/apiClient";
import { useStore } from "@/lib/store";
import { TaskStatus } from "@/lib/types";
import Link from "next/link";
import { useEffect, useState } from "react";

interface TaskDetailProps {
  taskId: string;
}

const TaskDetail = ({ taskId }: TaskDetailProps) => {
  const { selectedTask: task, fetchTask, isLoading, error } = useStore();
  const [completeLoading, setCompleteLoading] = useState(false);
  const [completeError, setCompleteError] = useState<string | null>(null);

  useEffect(() => {
    fetchTask(taskId);
  }, [fetchTask, taskId]);

  const handleCompleteTask = async () => {
    if (!task) return;
    setCompleteLoading(true);
    setCompleteError(null);
    try {
      await api.updateTaskStatus(task.id, "completed" as TaskStatus);
      // Refetch the task to update the UI
      await fetchTask(taskId);
    } catch (err) {
      // Type-safe error handling
      if (err instanceof ApiError) {
        setCompleteError(err.message);
      } else if (err instanceof Error) {
        setCompleteError(err.message);
      } else {
        setCompleteError("Failed to update task status.");
      }
    } finally {
      setCompleteLoading(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-1/2 mt-2" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </CardContent>
        <CardFooter>
          <Skeleton className="h-10 w-24" />
        </CardFooter>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Error</CardTitle>
        </CardHeader>
        <CardContent>
          <div role="alert" className="text-red-500">
            {error instanceof ApiError
              ? error.message
              : "An unknown error occurred"}
          </div>
        </CardContent>
        <CardFooter>
          <Button asChild variant="outline">
            <Link href="/tasks">Back to Tasks</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  if (!task) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Task Not Found</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            The requested task could not be found.
          </p>
        </CardContent>
        <CardFooter>
          <Button asChild variant="outline">
            <Link href="/tasks">Back to Tasks</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl">{task.description}</CardTitle>
            <CardDescription>Task ID: {task.id}</CardDescription>
          </div>
          <Badge
            variant={task.status === "completed" ? "success" : "default"}
            className="ml-4"
          >
            {task.status}
          </Badge>
        </div>
        {completeError && (
          <div role="alert" className="text-red-500 mt-2">
            {completeError}
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold mb-2">Details</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Crew ID</p>
              <p>{task.crew_id}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Created</p>
              <p>{new Date(task.created_at).toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Last Updated</p>
              <p>{new Date(task.updated_at).toLocaleString()}</p>
            </div>
            {task.completed_at && (
              <div>
                <p className="text-sm text-muted-foreground">Completed</p>
                <p>{new Date(task.completed_at).toLocaleString()}</p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button asChild variant="outline">
          <Link href="/tasks">Back to Tasks</Link>
        </Button>
        {task.status !== "completed" && (
          <Button
            variant="default"
            onClick={handleCompleteTask}
            disabled={completeLoading}
            aria-label="Mark task as completed"
          >
            {completeLoading ? "Completing..." : "Mark as Completed"}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
};

export default TaskDetail;
