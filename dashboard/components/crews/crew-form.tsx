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
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, Save, Plus, X, Users, ListTodo } from "lucide-react";
import { useManagementStore } from "@/stores/management-store";
import type { CrewConfig, CreateCrewRequest, UpdateCrewRequest } from "@/lib/api/management";

interface CrewFormProps {
  crew?: CrewConfig;
  onSuccess?: (crew: CrewConfig) => void;
  onCancel?: () => void;
}

export function CrewForm({ crew, onSuccess, onCancel }: CrewFormProps) {
  const {
    createCrew,
    updateCrew,
    crewsLoading,
    agents,
    fetchAgents,
    tasks,
    fetchTasks,
  } = useManagementStore();

  const isEditing = !!crew;

  // Form state
  const [name, setName] = useState(crew?.name || "");
  const [description, setDescription] = useState(crew?.description || "");
  const [process, setProcess] = useState<"sequential" | "hierarchical">(
    crew?.process || "sequential"
  );
  const [selectedAgentIds, setSelectedAgentIds] = useState<string[]>(crew?.agentIds || []);
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>(crew?.taskIds || []);
  const [managerAgentId, setManagerAgentId] = useState(crew?.managerAgentId || "");
  const [managerLlm, setManagerLlm] = useState(crew?.managerLlm || "");
  const [verbose, setVerbose] = useState(crew?.verbose ?? false);
  const [memory, setMemory] = useState(crew?.memory ?? false);
  const [cache, setCache] = useState(crew?.cache ?? true);
  const [stream, setStream] = useState(crew?.stream ?? true);
  const [maxRpm, setMaxRpm] = useState<number | undefined>(crew?.maxRpm);
  const [tags, setTags] = useState<string[]>(crew?.tags || []);
  const [newTag, setNewTag] = useState("");

  // Load agents and tasks on mount
  useEffect(() => {
    fetchAgents();
    fetchTasks();
  }, [fetchAgents, fetchTasks]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      if (isEditing && crew) {
        const updates: UpdateCrewRequest = {
          name,
          description,
          process,
          agentIds: selectedAgentIds,
          taskIds: selectedTaskIds,
          managerAgentId: managerAgentId || undefined,
          managerLlm: managerLlm || undefined,
          verbose,
          memory,
          cache,
          stream,
          maxRpm,
          tags,
        };
        const updated = await updateCrew(crew.id, updates);
        onSuccess?.(updated);
      } else {
        const data: CreateCrewRequest = {
          name,
          description,
          process,
          agentIds: selectedAgentIds,
          taskIds: selectedTaskIds,
          managerAgentId: managerAgentId || undefined,
          managerLlm: managerLlm || undefined,
          verbose,
          memory,
          cache,
          stream,
          maxRpm,
          tags,
        };
        const created = await createCrew(data);
        onSuccess?.(created);
      }
    } catch (error) {
      console.error("Failed to save crew:", error);
    }
  };

  const toggleAgent = (agentId: string) => {
    if (selectedAgentIds.includes(agentId)) {
      setSelectedAgentIds(selectedAgentIds.filter((id) => id !== agentId));
    } else {
      setSelectedAgentIds([...selectedAgentIds, agentId]);
    }
  };

  const toggleTask = (taskId: string) => {
    if (selectedTaskIds.includes(taskId)) {
      setSelectedTaskIds(selectedTaskIds.filter((id) => id !== taskId));
    } else {
      setSelectedTaskIds([...selectedTaskIds, taskId]);
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
      <Card className="trading-card">
        <CardHeader>
          <CardTitle>{isEditing ? "Edit Crew" : "Create Crew"}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Basic Info */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="name">Crew Name *</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Research Crew"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="process">Process Type</Label>
              <Select value={process} onValueChange={(v) => setProcess(v as "sequential" | "hierarchical")}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sequential">Sequential</SelectItem>
                  <SelectItem value="hierarchical">Hierarchical</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this crew do?"
              rows={2}
            />
          </div>

          {/* Hierarchical Settings */}
          {process === "hierarchical" && (
            <div className="grid gap-4 md:grid-cols-2 p-4 border rounded-lg bg-muted/30">
              <div className="space-y-2">
                <Label htmlFor="managerAgent">Manager Agent</Label>
                <Select value={managerAgentId} onValueChange={setManagerAgentId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select manager" />
                  </SelectTrigger>
                  <SelectContent>
                    {agents.map((agent) => (
                      <SelectItem key={agent.id} value={agent.id}>
                        {agent.role}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="managerLlm">Manager LLM</Label>
                <Input
                  id="managerLlm"
                  value={managerLlm}
                  onChange={(e) => setManagerLlm(e.target.value)}
                  placeholder="e.g., gpt-4"
                />
              </div>
            </div>
          )}

          {/* Agent Selection */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              Agents ({selectedAgentIds.length} selected)
            </Label>
            <div className="max-h-48 overflow-y-auto border rounded-lg p-2 space-y-2">
              {agents.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No agents available. Create agents first.
                </p>
              ) : (
                agents.map((agent) => (
                  <div
                    key={agent.id}
                    className="flex items-center space-x-2 p-2 hover:bg-muted rounded"
                  >
                    <Checkbox
                      id={`agent-${agent.id}`}
                      checked={selectedAgentIds.includes(agent.id)}
                      onCheckedChange={() => toggleAgent(agent.id)}
                    />
                    <label
                      htmlFor={`agent-${agent.id}`}
                      className="text-sm cursor-pointer flex-1"
                    >
                      {agent.role}
                    </label>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Task Selection */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <ListTodo className="h-4 w-4" />
              Tasks ({selectedTaskIds.length} selected)
            </Label>
            <div className="max-h-48 overflow-y-auto border rounded-lg p-2 space-y-2">
              {tasks.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No tasks available. Create tasks first.
                </p>
              ) : (
                tasks.map((task) => (
                  <div
                    key={task.id}
                    className="flex items-center space-x-2 p-2 hover:bg-muted rounded"
                  >
                    <Checkbox
                      id={`task-${task.id}`}
                      checked={selectedTaskIds.includes(task.id)}
                      onCheckedChange={() => toggleTask(task.id)}
                    />
                    <label
                      htmlFor={`task-${task.id}`}
                      className="text-sm cursor-pointer flex-1 truncate"
                    >
                      {task.name || task.description.slice(0, 50)}
                    </label>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Settings */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="maxRpm">Max RPM (optional)</Label>
              <Input
                id="maxRpm"
                type="number"
                value={maxRpm ?? ""}
                onChange={(e) => setMaxRpm(e.target.value ? parseInt(e.target.value) : undefined)}
                min={1}
                placeholder="No limit"
              />
            </div>
          </div>

          {/* Behavior Flags */}
          <div className="space-y-4">
            <Label>Behavior Settings</Label>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="flex items-center space-x-2">
                <Switch id="verbose" checked={verbose} onCheckedChange={setVerbose} />
                <Label htmlFor="verbose" className="font-normal">
                  Verbose
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch id="memory" checked={memory} onCheckedChange={setMemory} />
                <Label htmlFor="memory" className="font-normal">
                  Memory
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch id="cache" checked={cache} onCheckedChange={setCache} />
                <Label htmlFor="cache" className="font-normal">
                  Cache
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch id="stream" checked={stream} onCheckedChange={setStream} />
                <Label htmlFor="stream" className="font-normal">
                  Stream
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

        <CardFooter className="flex justify-end gap-2">
          {onCancel && (
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button type="submit" disabled={crewsLoading || !name}>
            {crewsLoading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isEditing ? "Save Changes" : "Create Crew"}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}
