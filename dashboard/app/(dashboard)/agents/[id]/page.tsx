"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  Edit,
  Trash2,
  Bot,
  Target,
  BookOpen,
  Wrench,
  Settings,
  Clock,
} from "lucide-react";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import { AgentForm } from "@/components/agents/agent-form";
import { useManagementStore } from "@/stores/management-store";

export default function AgentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const agentId = params.id as string;

  const {
    selectedAgent: agent,
    fetchAgent,
    deleteAgent,
    agentsLoading,
    agentsError,
    tools,
    fetchTools,
  } = useManagementStore();

  const [isEditing, setIsEditing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (agentId) {
      fetchAgent(agentId);
      fetchTools();
    }
  }, [agentId, fetchAgent, fetchTools]);

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await deleteAgent(agentId);
      router.push("/agents");
    } catch (error) {
      console.error("Failed to delete agent:", error);
      setIsDeleting(false);
    }
  };

  if (agentsLoading && !agent) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading agent...</p>
        </div>
      </div>
    );
  }

  if (agentsError || !agent) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-red-500" />
          <div>
            <p className="font-medium">Agent not found</p>
            <p className="text-sm text-muted-foreground mt-1">
              {agentsError || "The agent you're looking for doesn't exist."}
            </p>
          </div>
          <Button onClick={() => router.push("/agents")} variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Agents
          </Button>
        </div>
      </div>
    );
  }

  const agentTools = tools.filter((t) => agent.toolIds?.includes(t.id));

  if (isEditing) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => setIsEditing(false)}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h2 className="text-lg font-semibold">Edit Agent</h2>
        </div>
        <AgentForm
          agent={agent}
          onSuccess={() => {
            setIsEditing(false);
            fetchAgent(agentId);
          }}
          onCancel={() => setIsEditing(false)}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.push("/agents")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-3">
            <AgentAvatar name={agent.role} desk="spot" size="lg" />
            <div>
              <h2 className="text-lg font-semibold">{agent.name || agent.role}</h2>
              <p className="text-sm text-muted-foreground">{agent.role}</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => setIsEditing(true)}>
            <Edit className="h-4 w-4 mr-2" />
            Edit
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive">
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete Agent?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete the agent "{agent.role}". This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDelete}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  disabled={isDeleting}
                >
                  {isDeleting ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <Trash2 className="h-4 w-4 mr-2" />
                  )}
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* Tags */}
      {agent.tags && agent.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {agent.tags.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>
      )}

      {/* Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="tools">Tools ({agentTools.length})</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Goal */}
            <Card className="trading-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Goal
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{agent.goal}</p>
              </CardContent>
            </Card>

            {/* Backstory */}
            <Card className="trading-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  Backstory
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{agent.backstory}</p>
              </CardContent>
            </Card>
          </div>

          {/* Stats */}
          <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
            <Card className="trading-card">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Bot className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm font-medium">{agent.llm || "Default LLM"}</p>
                  <p className="text-xs text-muted-foreground">Model</p>
                </div>
              </CardContent>
            </Card>

            <Card className="trading-card">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Settings className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm font-medium">{agent.maxIter}</p>
                  <p className="text-xs text-muted-foreground">Max Iterations</p>
                </div>
              </CardContent>
            </Card>

            <Card className="trading-card">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Wrench className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm font-medium">{agentTools.length}</p>
                  <p className="text-xs text-muted-foreground">Tools</p>
                </div>
              </CardContent>
            </Card>

            <Card className="trading-card">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Clock className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm font-medium">
                    {new Date(agent.createdAt).toLocaleDateString()}
                  </p>
                  <p className="text-xs text-muted-foreground">Created</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="tools" className="space-y-4">
          {agentTools.length === 0 ? (
            <Card className="trading-card">
              <CardContent className="py-8 text-center">
                <Wrench className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No tools assigned to this agent.
                </p>
                <Button variant="outline" className="mt-4" onClick={() => setIsEditing(true)}>
                  Assign Tools
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {agentTools.map((tool) => (
                <Card key={tool.id} className="trading-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">{tool.name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {tool.description || "No description"}
                    </p>
                    <Badge variant="outline" className="mt-2">
                      {tool.toolType}
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="text-sm font-medium">Behavior Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Verbose Logging</p>
                  <p className="text-xs text-muted-foreground">
                    Enable detailed logging during execution
                  </p>
                </div>
                <Badge variant={agent.verbose ? "default" : "secondary"}>
                  {agent.verbose ? "Enabled" : "Disabled"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Allow Delegation</p>
                  <p className="text-xs text-muted-foreground">
                    Allow this agent to delegate tasks to others
                  </p>
                </div>
                <Badge variant={agent.allowDelegation ? "default" : "secondary"}>
                  {agent.allowDelegation ? "Enabled" : "Disabled"}
                </Badge>
              </div>
              {agent.maxRpm && (
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Max RPM</p>
                    <p className="text-xs text-muted-foreground">
                      Maximum requests per minute
                    </p>
                  </div>
                  <Badge variant="outline">{agent.maxRpm}</Badge>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="text-sm font-medium">Metadata</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">ID</span>
                <code className="bg-muted px-2 py-0.5 rounded text-xs">{agent.id}</code>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Created</span>
                <span>{new Date(agent.createdAt).toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Updated</span>
                <span>{new Date(agent.updatedAt).toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
