"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2, Save, Plus, X } from "lucide-react";
import { useManagementStore } from "@/stores/management-store";
import type { TaskConfig, CreateTaskRequest, UpdateTaskRequest } from "@/lib/api/management";

interface TaskFormProps {
  task?: TaskConfig;
  onSuccess?: (task: TaskConfig) => void;
  onCancel?: () => void;
}

export function TaskForm({ task, onSuccess, onCancel }: TaskFormProps) {
  const {
    createTask,
    updateTask,
    tasksLoading,
    agents,
    fetchAgents,
  } = useManagementStore();

  const isEditing = !!task;

  // Form state
  const [name, setName] = useState(task?.name || "");
  const [description, setDescription] = useState(task?.description || "");
  const [expectedOutput, setExpectedOutput] = useState(task?.expectedOutput || "");
  const [agentId, setAgentId] = useState(task?.agentId || "");
  const [asyncExecution, setAsyncExecution] = useState(task?.asyncExecution ?? false);
  const [humanInput, setHumanInput] = useState(task?.humanInput ?? false);
  const [actionBased, setActionBased] = useState(task?.actionBased ?? true);
  const [tags, setTags] = useState<string[]>(task?.tags || []);
  const [newTag, setNewTag] = useState("");

  // Load agents on mount
  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      if (isEditing && task) {
        const updates: UpdateTaskRequest = {
          name: name || undefined,
          description,
          expectedOutput,
          agentId: agentId || undefined,
          asyncExecution,
          humanInput,
          actionBased,
          tags,
        };
        const updated = await updateTask(task.id, updates);
        onSuccess?.(updated);
      } else {
        const data: CreateTaskRequest = {
          name: name || undefined,
          description,
          expectedOutput,
          agentId: agentId || undefined,
          asyncExecution,
          humanInput,
          actionBased,
          tags,
        };
        const created = await createTask(data);
        onSuccess?.(created);
      }
    } catch (error) {
      console.error("Failed to save task:", error);
    }
  };

  const addTag = () => {
    if (newTag && !tags.includes(newTag)) {
      setTags([...tags, newTag]);
      setNewTag("");
    }
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter((t) => t !== tag));
  };

  return (
    <form onSubmit={handleSubmit}>
      <Card className="trading-card border-0 shadow-none">
        <CardContent className="space-y-4 px-0">
          {/* Basic Info */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="name">Task Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Market Analysis"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="agent">Assigned Agent</Label>
              <Select value={agentId} onValueChange={setAgentId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select agent" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Unassigned</SelectItem>
                  {agents.map((agent) => (
                    <SelectItem key={agent.id} value={agent.id}>
                      {agent.role}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this task should accomplish..."
              rows={3}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="expectedOutput">Expected Output</Label>
            <Textarea
              id="expectedOutput"
              value={expectedOutput}
              onChange={(e) => setExpectedOutput(e.target.value)}
              placeholder="What output format is expected?"
              rows={2}
            />
          </div>

          {/* Behavior Flags */}
          <div className="space-y-4">
            <Label>Behavior Settings</Label>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="flex items-center space-x-2">
                <Switch
                  id="actionBased"
                  checked={actionBased}
                  onCheckedChange={setActionBased}
                />
                <Label htmlFor="actionBased" className="font-normal">
                  Action-based
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="asyncExecution"
                  checked={asyncExecution}
                  onCheckedChange={setAsyncExecution}
                />
                <Label htmlFor="asyncExecution" className="font-normal">
                  Async Execution
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="humanInput"
                  checked={humanInput}
                  onCheckedChange={setHumanInput}
                />
                <Label htmlFor="humanInput" className="font-normal">
                  Human Input
                </Label>
              </div>
            </div>
          </div>

          {/* Tags */}
          <div className="space-y-2">
            <Label>Tags</Label>
            <div className="flex gap-2">
              <Input
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                placeholder="Add tag..."
                onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addTag())}
              />
              <Button type="button" variant="outline" onClick={addTag}>
                <Plus className="h-4 w-4" />
              </Button>
            </div>
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="gap-1">
                    {tag}
                    <button
                      type="button"
                      onClick={() => removeTag(tag)}
                      className="ml-1 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex justify-end gap-2 px-0">
          {onCancel && (
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button type="submit" disabled={tasksLoading || !description}>
            {tasksLoading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isEditing ? "Save Changes" : "Create Task"}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}
