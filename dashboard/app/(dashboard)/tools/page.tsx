"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Plus,
  Wrench,
  Loader2,
  RefreshCw,
  AlertCircle,
  Trash2,
  Edit,
  Save,
  Package,
} from "lucide-react";
import { useManagementStore } from "@/stores/management-store";
import { toolApi } from "@/lib/api/management";
import type { ToolConfig, CreateToolRequest, UpdateToolRequest, BuiltinTool } from "@/lib/api/management";

export default function ToolsPage() {
  const {
    tools,
    toolsTotal,
    toolsLoading,
    toolsError,
    fetchTools,
    createTool,
    updateTool,
    deleteTool,
    selectedTool,
    selectTool,
  } = useManagementStore();

  const [search, setSearch] = useState("");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [builtinTools, setBuiltinTools] = useState<BuiltinTool[]>([]);

  // Form state
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formToolType, setFormToolType] = useState("builtin");
  const [formClassName, setFormClassName] = useState("");
  const [formModulePath, setFormModulePath] = useState("");
  const [formTags, setFormTags] = useState("");

  useEffect(() => {
    fetchTools();
    loadBuiltinTools();
  }, [fetchTools]);

  const loadBuiltinTools = async () => {
    try {
      const tools = await toolApi.listBuiltin();
      setBuiltinTools(tools);
    } catch (error) {
      console.error("Failed to load builtin tools:", error);
    }
  };

  const resetForm = () => {
    setFormName("");
    setFormDescription("");
    setFormToolType("builtin");
    setFormClassName("");
    setFormModulePath("");
    setFormTags("");
  };

  const openEditDialog = (tool: ToolConfig) => {
    selectTool(tool);
    setFormName(tool.name);
    setFormDescription(tool.description || "");
    setFormToolType(tool.toolType || "builtin");
    setFormClassName(tool.className || "");
    setFormModulePath(tool.modulePath || "");
    setFormTags(tool.tags?.join(", ") || "");
    setIsEditDialogOpen(true);
  };

  const handleCreate = async () => {
    const data: CreateToolRequest = {
      name: formName,
      description: formDescription,
      toolType: formToolType,
      className: formClassName || undefined,
      modulePath: formModulePath || undefined,
      tags: formTags ? formTags.split(",").map((t) => t.trim()) : [],
    };

    try {
      await createTool(data);
      setIsCreateDialogOpen(false);
      resetForm();
    } catch (error) {
      console.error("Failed to create tool:", error);
    }
  };

  const handleUpdate = async () => {
    if (!selectedTool) return;

    const data: UpdateToolRequest = {
      name: formName,
      description: formDescription,
      toolType: formToolType,
      className: formClassName || undefined,
      modulePath: formModulePath || undefined,
      tags: formTags ? formTags.split(",").map((t) => t.trim()) : [],
    };

    try {
      await updateTool(selectedTool.id, data);
      setIsEditDialogOpen(false);
      selectTool(null);
      resetForm();
    } catch (error) {
      console.error("Failed to update tool:", error);
    }
  };

  const handleDelete = async (toolId: string) => {
    try {
      await deleteTool(toolId);
    } catch (error) {
      console.error("Failed to delete tool:", error);
    }
  };

  const handleRegisterBuiltin = async (builtin: BuiltinTool) => {
    const data: CreateToolRequest = {
      name: builtin.name,
      description: builtin.description,
      toolType: "crewai_tools",
      className: builtin.name,
      modulePath: builtin.module,
    };

    try {
      await createTool(data);
    } catch (error) {
      console.error("Failed to register builtin tool:", error);
    }
  };

  const filteredTools = search
    ? tools.filter(
        (tool) =>
          tool.name.toLowerCase().includes(search.toLowerCase()) ||
          tool.description?.toLowerCase().includes(search.toLowerCase())
      )
    : tools;

  const registeredToolNames = new Set(tools.map((t) => t.name));

  if (toolsLoading && tools.length === 0) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading tools...</p>
        </div>
      </div>
    );
  }

  if (toolsError) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-red-500" />
          <div>
            <p className="font-medium">Failed to load tools</p>
            <p className="text-sm text-muted-foreground mt-1">{toolsError}</p>
          </div>
          <Button onClick={() => fetchTools()} variant="outline" className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const ToolForm = ({ isEdit = false }: { isEdit?: boolean }) => (
    <div className="space-y-4 py-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="name">Tool Name *</Label>
          <Input
            id="name"
            value={formName}
            onChange={(e) => setFormName(e.target.value)}
            placeholder="e.g., SerperDevTool"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="toolType">Tool Type</Label>
          <Select value={formToolType} onValueChange={setFormToolType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="builtin">Built-in</SelectItem>
              <SelectItem value="crewai_tools">CrewAI Tools</SelectItem>
              <SelectItem value="custom">Custom</SelectItem>
              <SelectItem value="mcp">MCP Server</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={formDescription}
          onChange={(e) => setFormDescription(e.target.value)}
          placeholder="What does this tool do?"
          rows={2}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="className">Class Name</Label>
          <Input
            id="className"
            value={formClassName}
            onChange={(e) => setFormClassName(e.target.value)}
            placeholder="e.g., SerperDevTool"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="modulePath">Module Path</Label>
          <Input
            id="modulePath"
            value={formModulePath}
            onChange={(e) => setFormModulePath(e.target.value)}
            placeholder="e.g., crewai_tools"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="tags">Tags (comma-separated)</Label>
        <Input
          id="tags"
          value={formTags}
          onChange={(e) => setFormTags(e.target.value)}
          placeholder="search, web, api"
        />
      </div>

      <div className="flex justify-end gap-2 pt-4">
        <Button
          type="button"
          variant="outline"
          onClick={() => {
            isEdit ? setIsEditDialogOpen(false) : setIsCreateDialogOpen(false);
            resetForm();
          }}
        >
          Cancel
        </Button>
        <Button onClick={isEdit ? handleUpdate : handleCreate} disabled={!formName}>
          <Save className="h-4 w-4 mr-2" />
          {isEdit ? "Save Changes" : "Create Tool"}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Wrench className="h-5 w-5" />
            Tools
          </h2>
          <p className="text-sm text-muted-foreground terminal-text">
            {toolsTotal} tools registered
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Register Tool
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Register New Tool</DialogTitle>
              </DialogHeader>
              <ToolForm />
            </DialogContent>
          </Dialog>
          <Button onClick={() => fetchTools()} variant="ghost" size="icon" className="h-9 w-9">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search tools..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 pl-8"
            />
          </div>
        </div>
      </div>

      {/* Registered Tools */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium">Registered Tools</h3>
        {filteredTools.length === 0 ? (
          <Card className="trading-card">
            <CardContent className="py-8 text-center">
              <Wrench className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                {search ? `No tools found matching "${search}"` : "No tools registered yet."}
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredTools.map((tool) => (
              <Card key={tool.id} className="trading-card">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-base font-medium">{tool.name}</CardTitle>
                    <Badge variant="outline">{tool.toolType}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {tool.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {tool.description}
                    </p>
                  )}

                  {tool.className && (
                    <div className="text-xs">
                      <span className="text-muted-foreground">Class: </span>
                      <code className="bg-muted px-1 py-0.5 rounded">{tool.className}</code>
                    </div>
                  )}

                  {tool.tags && tool.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {tool.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}

                  <div className="flex gap-2 pt-2">
                    <Button size="sm" variant="outline" onClick={() => openEditDialog(tool)}>
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
                          <AlertDialogTitle>Delete Tool?</AlertDialogTitle>
                          <AlertDialogDescription>
                            This will unregister "{tool.name}". Any agents using this tool will
                            need to be updated.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => handleDelete(tool.id)}
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
      </div>

      {/* Builtin Tools Catalog */}
      {builtinTools.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium flex items-center gap-2">
            <Package className="h-4 w-4" />
            Available Built-in Tools
          </h3>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {builtinTools.map((builtin) => {
              const isRegistered = registeredToolNames.has(builtin.name);
              return (
                <Card key={builtin.name} className="trading-card">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium truncate">{builtin.name}</h4>
                        <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                          {builtin.description || "No description"}
                        </p>
                      </div>
                      <Button
                        size="sm"
                        variant={isRegistered ? "secondary" : "default"}
                        disabled={isRegistered}
                        onClick={() => handleRegisterBuiltin(builtin)}
                      >
                        {isRegistered ? "Added" : "Add"}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Edit Tool</DialogTitle>
          </DialogHeader>
          <ToolForm isEdit />
        </DialogContent>
      </Dialog>
    </div>
  );
}
