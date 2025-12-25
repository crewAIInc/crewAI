"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
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
import { Loader2, Save, X, Plus, Trash2 } from "lucide-react";
import { useManagementStore } from "@/stores/management-store";
import type { AgentConfig, CreateAgentRequest, UpdateAgentRequest } from "@/lib/api/management";

interface AgentFormProps {
  agent?: AgentConfig;
  onSuccess?: (agent: AgentConfig) => void;
  onCancel?: () => void;
}

const LLM_OPTIONS = [
  { value: "gpt-4", label: "GPT-4" },
  { value: "gpt-4o", label: "GPT-4o" },
  { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo" },
  { value: "claude-3-opus-20240229", label: "Claude 3 Opus" },
  { value: "claude-3-sonnet-20240229", label: "Claude 3 Sonnet" },
  { value: "claude-3-haiku-20240307", label: "Claude 3 Haiku" },
  { value: "anthropic/claude-3.5-sonnet", label: "Claude 3.5 Sonnet" },
  { value: "openai/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", label: "LM Studio (Local)" },
];

export function AgentForm({ agent, onSuccess, onCancel }: AgentFormProps) {
  const router = useRouter();
  const { createAgent, updateAgent, agentsLoading, fetchTools, tools } = useManagementStore();

  const isEditing = !!agent;

  // Form state
  const [role, setRole] = useState(agent?.role || "");
  const [goal, setGoal] = useState(agent?.goal || "");
  const [backstory, setBackstory] = useState(agent?.backstory || "");
  const [name, setName] = useState(agent?.name || "");
  const [llm, setLlm] = useState(agent?.llm || "");
  const [maxIter, setMaxIter] = useState(agent?.maxIter ?? 25);
  const [maxRpm, setMaxRpm] = useState<number | undefined>(agent?.maxRpm);
  const [verbose, setVerbose] = useState(agent?.verbose ?? false);
  const [allowDelegation, setAllowDelegation] = useState(agent?.allowDelegation ?? false);
  const [selectedToolIds, setSelectedToolIds] = useState<string[]>(agent?.toolIds || []);
  const [tags, setTags] = useState<string[]>(agent?.tags || []);
  const [newTag, setNewTag] = useState("");

  // Load tools on mount
  useEffect(() => {
    fetchTools();
  }, [fetchTools]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      if (isEditing && agent) {
        const updates: UpdateAgentRequest = {
          role,
          goal,
          backstory,
          name: name || undefined,
          llm: llm || undefined,
          maxIter,
          maxRpm,
          verbose,
          allowDelegation,
          toolIds: selectedToolIds,
          tags,
        };
        const updated = await updateAgent(agent.id, updates);
        onSuccess?.(updated);
      } else {
        const data: CreateAgentRequest = {
          role,
          goal,
          backstory,
          name: name || undefined,
          llm: llm || undefined,
          maxIter,
          maxRpm,
          verbose,
          allowDelegation,
          toolIds: selectedToolIds,
          tags,
        };
        const created = await createAgent(data);
        onSuccess?.(created);
      }
    } catch (error) {
      console.error("Failed to save agent:", error);
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

  const toggleTool = (toolId: string) => {
    if (selectedToolIds.includes(toolId)) {
      setSelectedToolIds(selectedToolIds.filter((id) => id !== toolId));
    } else {
      setSelectedToolIds([...selectedToolIds, toolId]);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Card className="trading-card">
        <CardHeader>
          <CardTitle>{isEditing ? "Edit Agent" : "Create Agent"}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Basic Info */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="role">Role *</Label>
              <Input
                id="role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                placeholder="e.g., Research Analyst"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="name">Display Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Optional display name"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="goal">Goal *</Label>
            <Textarea
              id="goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="What is the agent's objective?"
              rows={2}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="backstory">Backstory *</Label>
            <Textarea
              id="backstory"
              value={backstory}
              onChange={(e) => setBackstory(e.target.value)}
              placeholder="Background context for the agent's persona..."
              rows={3}
              required
            />
          </div>

          {/* LLM Configuration */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="llm">LLM Model</Label>
              <Select value={llm} onValueChange={setLlm}>
                <SelectTrigger>
                  <SelectValue placeholder="Select LLM model" />
                </SelectTrigger>
                <SelectContent>
                  {LLM_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="maxIter">Max Iterations</Label>
              <Input
                id="maxIter"
                type="number"
                value={maxIter}
                onChange={(e) => setMaxIter(parseInt(e.target.value) || 25)}
                min={1}
                max={100}
              />
            </div>
          </div>

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
            <div className="flex flex-wrap gap-6">
              <div className="flex items-center space-x-2">
                <Switch
                  id="verbose"
                  checked={verbose}
                  onCheckedChange={setVerbose}
                />
                <Label htmlFor="verbose" className="font-normal">
                  Verbose logging
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="allowDelegation"
                  checked={allowDelegation}
                  onCheckedChange={setAllowDelegation}
                />
                <Label htmlFor="allowDelegation" className="font-normal">
                  Allow delegation
                </Label>
              </div>
            </div>
          </div>

          {/* Tools */}
          {tools.length > 0 && (
            <div className="space-y-2">
              <Label>Available Tools</Label>
              <div className="flex flex-wrap gap-2">
                {tools.map((tool) => (
                  <Badge
                    key={tool.id}
                    variant={selectedToolIds.includes(tool.id) ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => toggleTool(tool.id)}
                  >
                    {tool.name}
                  </Badge>
                ))}
              </div>
            </div>
          )}

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
          <Button type="submit" disabled={agentsLoading || !role || !goal || !backstory}>
            {agentsLoading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isEditing ? "Save Changes" : "Create Agent"}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}
