"use client";

import { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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
  Kanban,
  LayoutList,
  Clock,
  CheckCircle2,
  Loader2,
  XCircle,
  Trash2,
  Plus,
  Search,
  RefreshCw,
  AlertCircle,
  Edit,
  Settings2,
  Zap,
  User,
} from "lucide-react";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import { useTaskStore } from "@/stores/task-store";
import { useManagementStore } from "@/stores/management-store";
import { ConnectionIndicator } from "@/components/layout/connection-status";
import { TaskForm } from "@/components/tasks/task-form";
import type { Task, TaskStatus } from "@/lib/types/task";
import type { TaskConfig } from "@/lib/api/management";
import type { Desk } from "@/lib/types/agent";

// Demo tasks for when store is empty
const demoTasks: Task[] = [
  { id: "1", name: "Market Analysis", description: "", agentId: "research-head", agentRole: "Research Head", status: "pending" },
  { id: "2", name: "Risk Assessment", description: "", agentId: "gcro", agentRole: "Group CRO", status: "pending" },
  { id: "3", name: "Funding Check", description: "", agentId: "carry-head", agentRole: "Carry Head", status: "pending" },
  { id: "4", name: "Portfolio Review", description: "", agentId: "cio-spot", agentRole: "CIO Spot", status: "running", progress: 75 },
  { id: "5", name: "Order Execution", description: "", agentId: "execution-head", agentRole: "Execution Head", status: "running", progress: 45 },
  { id: "6", name: "Morning Briefing", description: "", agentId: "ceo", agentRole: "CEO", status: "completed", duration: 2.3 },
  { id: "7", name: "Exposure Report", description: "", agentId: "risk-monitor", agentRole: "Risk Monitor", status: "completed", duration: 1.8 },
  { id: "8", name: "Strategy Update", description: "", agentId: "systematic-head", agentRole: "Systematic Head", status: "completed", duration: 3.1 },
];

// Map agent role to desk
function getDeskFromRole(role: string): Desk {
  const lowerRole = role.toLowerCase();
  if (lowerRole.includes("ceo") || lowerRole.includes("group") || lowerRole.includes("gcio") || lowerRole.includes("gcro") || lowerRole.includes("gcoo") || lowerRole.includes("gcfo")) {
    return "staff";
  }
  if (lowerRole.includes("futures") || lowerRole.includes("carry") || lowerRole.includes("microstructure")) {
    return "futures";
  }
  return "spot";
}

function TaskCard({ task, status }: { task: Task; status: TaskStatus }) {
  const desk = getDeskFromRole(task.agentRole);

  return (
    <div
      className={`rounded-lg border p-3 space-y-3 transition-all hover:shadow-md ${
        status === "running" ? "border-blue-500/50 bg-blue-500/5" : ""
      } ${status === "failed" ? "border-red-500/50 bg-red-500/5" : ""}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1 min-w-0 flex-1">
          <h4 className="text-sm font-medium truncate">{task.name}</h4>
          <div className="flex items-center gap-2">
            <AgentAvatar name={task.agentRole} desk={desk} size="sm" />
            <span className="text-xs text-muted-foreground truncate">{task.agentRole}</span>
          </div>
        </div>
        {status === "pending" && (
          <Badge variant="secondary" className="status-pending text-[10px] flex-shrink-0">
            <Clock className="h-3 w-3 mr-1" />
            Pending
          </Badge>
        )}
        {status === "running" && (
          <Badge variant="secondary" className="status-running text-[10px] flex-shrink-0">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Running
          </Badge>
        )}
        {status === "completed" && (
          <Badge variant="secondary" className="status-completed text-[10px] flex-shrink-0">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Done
          </Badge>
        )}
        {status === "failed" && (
          <Badge variant="secondary" className="bg-red-500/10 text-red-500 text-[10px] flex-shrink-0">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        )}
      </div>

      {status === "running" && task.progress !== undefined && task.progress > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Progress</span>
            <span>{task.progress}%</span>
          </div>
          <Progress value={task.progress} className="h-1.5" />
        </div>
      )}

      {status === "completed" && task.duration !== undefined && (
        <div className="text-xs text-muted-foreground">
          Completed in {task.duration.toFixed(1)}s
        </div>
      )}

      {status === "failed" && task.error && (
        <div className="text-xs text-red-500 truncate">
          {task.error}
        </div>
      )}
    </div>
  );
}

function KanbanView() {
  const { tasks, pendingTaskIds, runningTaskIds, completedTaskIds, clearCompletedTasks } = useTaskStore();

  const storeHasTasks = tasks.size > 0;

  const { pendingTasks, runningTasks, completedTasks, failedTasks } = useMemo(() => {
    if (storeHasTasks) {
      const pending = pendingTaskIds.map(id => tasks.get(id)).filter((t): t is Task => !!t);
      const running = runningTaskIds.map(id => tasks.get(id)).filter((t): t is Task => !!t);
      const completed = completedTaskIds
        .map(id => tasks.get(id))
        .filter((t): t is Task => !!t && t.status === "completed");
      const failed = completedTaskIds
        .map(id => tasks.get(id))
        .filter((t): t is Task => !!t && t.status === "failed");
      return { pendingTasks: pending, runningTasks: running, completedTasks: completed, failedTasks: failed };
    }

    return {
      pendingTasks: demoTasks.filter(t => t.status === "pending"),
      runningTasks: demoTasks.filter(t => t.status === "running"),
      completedTasks: demoTasks.filter(t => t.status === "completed"),
      failedTasks: [] as Task[],
    };
  }, [storeHasTasks, tasks, pendingTaskIds, runningTaskIds, completedTaskIds]);

  return (
    <>
      {!storeHasTasks && (
        <div className="text-xs text-muted-foreground bg-muted/50 rounded-lg p-3 mb-4">
          Showing demo data. Connect to a crew on the Chat page to see real-time tasks.
        </div>
      )}

      <div className="grid grid-cols-3 gap-6 h-[calc(100vh-18rem)]">
        {/* Pending Column */}
        <Card className="flex flex-col">
          <CardHeader className="flex-shrink-0 pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-amber-500" />
                Pending
              </CardTitle>
              <Badge variant="secondary">{pendingTasks.length}</Badge>
            </div>
          </CardHeader>
          <ScrollArea className="flex-1 px-4 pb-4">
            <div className="space-y-3">
              {pendingTasks.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No pending tasks
                </div>
              ) : (
                pendingTasks.map((task, index) => (
                  <div
                    key={task.id}
                    className="animate-scale-in"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <TaskCard task={task} status="pending" />
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </Card>

        {/* Running Column */}
        <Card className="flex flex-col border-blue-500/30">
          <CardHeader className="flex-shrink-0 pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                Running
              </CardTitle>
              <Badge variant="secondary" className="bg-blue-500/10 text-blue-500">
                {runningTasks.length}
              </Badge>
            </div>
          </CardHeader>
          <ScrollArea className="flex-1 px-4 pb-4">
            <div className="space-y-3">
              {runningTasks.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No running tasks
                </div>
              ) : (
                runningTasks.map((task, index) => (
                  <div
                    key={task.id}
                    className="animate-scale-in"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <TaskCard task={task} status="running" />
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </Card>

        {/* Completed Column */}
        <Card className="flex flex-col">
          <CardHeader className="flex-shrink-0 pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500" />
                Completed
              </CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="bg-green-500/10 text-green-500">
                  {completedTasks.length + failedTasks.length}
                </Badge>
                {(completedTasks.length > 0 || failedTasks.length > 0) && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                    onClick={clearCompletedTasks}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>
          <ScrollArea className="flex-1 px-4 pb-4">
            <div className="space-y-3">
              {completedTasks.length === 0 && failedTasks.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-8">
                  No completed tasks
                </div>
              ) : (
                <>
                  {failedTasks.map((task, index) => (
                    <div
                      key={task.id}
                      className="animate-scale-in"
                      style={{ animationDelay: `${index * 50}ms` }}
                    >
                      <TaskCard task={task} status="failed" />
                    </div>
                  ))}
                  {completedTasks.map((task, index) => (
                    <div
                      key={task.id}
                      className="animate-scale-in"
                      style={{ animationDelay: `${(failedTasks.length + index) * 50}ms` }}
                    >
                      <TaskCard task={task} status="completed" />
                    </div>
                  ))}
                </>
              )}
            </div>
          </ScrollArea>
        </Card>
      </div>
    </>
  );
}

function ConfigurationsView() {
  const {
    tasks: taskConfigs,
    tasksTotal,
    tasksLoading,
    tasksError,
    fetchTasks,
    deleteTask,
    selectTask,
    selectedTask,
    agents,
    fetchAgents,
  } = useManagementStore();

  const [search, setSearch] = useState("");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);

  useEffect(() => {
    fetchTasks();
    fetchAgents();
  }, [fetchTasks, fetchAgents]);

  const filteredTasks = search
    ? taskConfigs.filter(
        (task) =>
          task.name?.toLowerCase().includes(search.toLowerCase()) ||
          task.description?.toLowerCase().includes(search.toLowerCase())
      )
    : taskConfigs;

  const getAgentName = (agentId?: string) => {
    if (!agentId) return "Unassigned";
    const agent = agents.find((a) => a.id === agentId);
    return agent?.role || "Unknown Agent";
  };

  const handleEdit = (task: TaskConfig) => {
    selectTask(task);
    setIsEditDialogOpen(true);
  };

  const handleDelete = async (taskId: string) => {
    try {
      await deleteTask(taskId);
    } catch (error) {
      console.error("Failed to delete task:", error);
    }
  };

  if (tasksLoading && taskConfigs.length === 0) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading task configurations...</p>
        </div>
      </div>
    );
  }

  if (tasksError) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-red-500" />
          <div>
            <p className="font-medium">Failed to load task configurations</p>
            <p className="text-sm text-muted-foreground mt-1">{tasksError}</p>
          </div>
          <Button onClick={() => fetchTasks()} variant="outline" className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Actions Bar */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-muted-foreground terminal-text">
          {tasksTotal} task configurations
        </p>
        <div className="flex items-center gap-3">
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Create Task
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Create New Task</DialogTitle>
              </DialogHeader>
              <TaskForm
                onSuccess={() => {
                  setIsCreateDialogOpen(false);
                  fetchTasks();
                }}
                onCancel={() => setIsCreateDialogOpen(false)}
              />
            </DialogContent>
          </Dialog>
          <Button onClick={() => fetchTasks()} variant="ghost" size="icon" className="h-9 w-9">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search tasks..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 pl-8"
            />
          </div>
        </div>
      </div>

      {/* Task Configs Grid */}
      {filteredTasks.length === 0 ? (
        <Card className="trading-card">
          <CardContent className="py-12 text-center">
            <Settings2 className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {search ? `No tasks found matching "${search}"` : "No task configurations yet."}
            </p>
            {!search && (
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => setIsCreateDialogOpen(true)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Create your first task
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredTasks.map((task) => (
            <Card key={task.id} className="trading-card hover:shadow-lg transition-all">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <CardTitle className="text-base font-medium">
                    {task.name || "Unnamed Task"}
                  </CardTitle>
                  <div className="flex gap-1">
                    {task.actionBased && (
                      <Badge variant="outline" className="text-xs">
                        <Zap className="h-3 w-3 mr-1" />
                        Action
                      </Badge>
                    )}
                    {task.asyncExecution && (
                      <Badge variant="secondary" className="text-xs">Async</Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {task.description}
                </p>

                <div className="flex items-center gap-2 text-sm">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">
                    {getAgentName(task.agentId)}
                  </span>
                </div>

                {task.expectedOutput && (
                  <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                    <span className="font-medium">Expected: </span>
                    <span className="line-clamp-1">{task.expectedOutput}</span>
                  </div>
                )}

                {task.tags && task.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {task.tags.slice(0, 3).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                    {task.tags.length > 3 && (
                      <Badge variant="secondary" className="text-xs">
                        +{task.tags.length - 3}
                      </Badge>
                    )}
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button size="sm" variant="outline" onClick={() => handleEdit(task)}>
                    <Edit className="h-4 w-4" />
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button size="sm" variant="outline" className="text-destructive">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete Task?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete "{task.name || "this task"}". Any crews using this task will need to be updated.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => handleDelete(task.id)}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Edit Task</DialogTitle>
          </DialogHeader>
          {selectedTask && (
            <TaskForm
              task={selectedTask}
              onSuccess={() => {
                setIsEditDialogOpen(false);
                selectTask(null);
                fetchTasks();
              }}
              onCancel={() => {
                setIsEditDialogOpen(false);
                selectTask(null);
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function TasksPage() {
  const [activeTab, setActiveTab] = useState("kanban");
  const { tasks } = useTaskStore();
  const { tasks: taskConfigs } = useManagementStore();

  const storeHasTasks = tasks.size > 0;
  const totalRealtime = storeHasTasks ? tasks.size : demoTasks.length;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Kanban className="h-5 w-5" />
            Tasks
          </h2>
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <span>
              {activeTab === "kanban"
                ? `${totalRealtime} real-time tasks`
                : `${taskConfigs.length} configurations`}
            </span>
            {activeTab === "kanban" && <ConnectionIndicator />}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="kanban" className="gap-2">
            <Kanban className="h-4 w-4" />
            Live Board
          </TabsTrigger>
          <TabsTrigger value="configurations" className="gap-2">
            <Settings2 className="h-4 w-4" />
            Configurations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="kanban" className="mt-4">
          <KanbanView />
        </TabsContent>

        <TabsContent value="configurations" className="mt-4">
          <ConfigurationsView />
        </TabsContent>
      </Tabs>
    </div>
  );
}
