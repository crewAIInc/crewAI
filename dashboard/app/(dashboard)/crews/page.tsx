"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  Search,
  Plus,
  Users,
  ListTodo,
  Loader2,
  RefreshCw,
  AlertCircle,
  Play,
  Settings,
  Trash2,
  Edit,
} from "lucide-react";
import { useManagementStore } from "@/stores/management-store";
import { CrewForm } from "@/components/crews/crew-form";
import { crewApi } from "@/lib/api/management";
import type { CrewConfig } from "@/lib/api/management";

export default function CrewsPage() {
  const router = useRouter();
  const {
    crews,
    crewsTotal,
    crewsLoading,
    crewsError,
    fetchCrews,
    deleteCrew,
    selectCrew,
    selectedCrew,
  } = useManagementStore();

  const [search, setSearch] = useState("");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isExecuting, setIsExecuting] = useState<string | null>(null);

  useEffect(() => {
    fetchCrews();
  }, [fetchCrews]);

  const filteredCrews = search
    ? crews.filter(
        (crew) =>
          crew.name.toLowerCase().includes(search.toLowerCase()) ||
          crew.description?.toLowerCase().includes(search.toLowerCase())
      )
    : crews;

  const handleExecute = async (crewId: string) => {
    setIsExecuting(crewId);
    try {
      const result = await crewApi.execute(crewId, {});
      console.log("Crew execution result:", result);
      // Could show a toast or notification here
    } catch (error) {
      console.error("Failed to execute crew:", error);
    } finally {
      setIsExecuting(null);
    }
  };

  const handleDelete = async (crewId: string) => {
    try {
      await deleteCrew(crewId);
    } catch (error) {
      console.error("Failed to delete crew:", error);
    }
  };

  const handleEdit = (crew: CrewConfig) => {
    selectCrew(crew);
    setIsEditDialogOpen(true);
  };

  if (crewsLoading && crews.length === 0) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading crews...</p>
        </div>
      </div>
    );
  }

  if (crewsError) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-red-500" />
          <div>
            <p className="font-medium">Failed to load crews</p>
            <p className="text-sm text-muted-foreground mt-1">{crewsError}</p>
          </div>
          <Button onClick={() => fetchCrews()} variant="outline" className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Users className="h-5 w-5" />
            Crews
          </h2>
          <p className="text-sm text-muted-foreground terminal-text">
            {crewsTotal} crews configured
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Create Crew
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Create New Crew</DialogTitle>
              </DialogHeader>
              <CrewForm
                onSuccess={() => {
                  setIsCreateDialogOpen(false);
                  fetchCrews();
                }}
                onCancel={() => setIsCreateDialogOpen(false)}
              />
            </DialogContent>
          </Dialog>
          <Button onClick={() => fetchCrews()} variant="ghost" size="icon" className="h-9 w-9">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search crews..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 pl-8"
            />
          </div>
        </div>
      </div>

      {/* Crew Grid */}
      {filteredCrews.length === 0 ? (
        <Card className="trading-card">
          <CardContent className="py-12 text-center">
            <Users className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {search ? `No crews found matching "${search}"` : "No crews created yet."}
            </p>
            {!search && (
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => setIsCreateDialogOpen(true)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Create your first crew
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredCrews.map((crew) => (
            <Card key={crew.id} className="trading-card hover:shadow-lg transition-all">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <CardTitle className="text-base font-medium">{crew.name}</CardTitle>
                  <Badge variant="outline">{crew.process}</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {crew.description && (
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {crew.description}
                  </p>
                )}

                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-1.5">
                    <Users className="h-4 w-4 text-muted-foreground" />
                    <span>{crew.agentIds?.length || 0} agents</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <ListTodo className="h-4 w-4 text-muted-foreground" />
                    <span>{crew.taskIds?.length || 0} tasks</span>
                  </div>
                </div>

                {crew.tags && crew.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {crew.tags.slice(0, 3).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                    {crew.tags.length > 3 && (
                      <Badge variant="secondary" className="text-xs">
                        +{crew.tags.length - 3}
                      </Badge>
                    )}
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={() => handleExecute(crew.id)}
                    disabled={isExecuting === crew.id}
                  >
                    {isExecuting === crew.id ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-1" />
                    ) : (
                      <Play className="h-4 w-4 mr-1" />
                    )}
                    Run
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => handleEdit(crew)}>
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
                        <AlertDialogTitle>Delete Crew?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete "{crew.name}". This action cannot be undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => handleDelete(crew.id)}
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
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Crew</DialogTitle>
          </DialogHeader>
          {selectedCrew && (
            <CrewForm
              crew={selectedCrew}
              onSuccess={() => {
                setIsEditDialogOpen(false);
                selectCrew(null);
                fetchCrews();
              }}
              onCancel={() => {
                setIsEditDialogOpen(false);
                selectCrew(null);
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
