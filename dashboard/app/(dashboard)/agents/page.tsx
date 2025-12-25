"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Search,
  LayoutGrid,
  List,
  Users,
  Loader2,
  RefreshCw,
  AlertCircle,
  Plus,
} from "lucide-react";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import { cn } from "@/lib/utils";

interface Agent {
  id: string;
  role: string;
  goal: string;
  backstory: string;
  crew: "staff" | "spot" | "futures";
  status: "idle" | "active" | "error";
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function AgentCard({
  agent,
  onClick,
}: {
  agent: Agent;
  onClick?: () => void;
}) {
  const desk = agent.crew;
  const shortName = agent.role.split(" ").slice(0, 2).join(" ");

  return (
    <Card
      className="group trading-card hover:shadow-lg transition-all cursor-pointer"
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className="relative">
            <AgentAvatar name={shortName} desk={desk} size="lg" />
            <div
              className={cn(
                "absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-card",
                agent.status === "active"
                  ? "bg-emerald-500 animate-pulse"
                  : agent.status === "error"
                  ? "bg-red-500"
                  : "bg-gray-400"
              )}
            />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-sm">{agent.role}</h3>
              <Badge
                variant="secondary"
                className={cn(
                  "text-[10px]",
                  desk === "staff"
                    ? "desk-staff"
                    : desk === "spot"
                    ? "desk-spot"
                    : "desk-futures"
                )}
              >
                {desk.toUpperCase()}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2">
              {agent.goal}
            </p>
            <div className="flex items-center gap-1 pt-1">
              <span
                className={cn(
                  "text-[10px] font-medium terminal-text",
                  agent.status === "active"
                    ? "text-emerald-400"
                    : agent.status === "error"
                    ? "text-red-400"
                    : "text-muted-foreground"
                )}
              >
                {agent.status === "active"
                  ? "Active"
                  : agent.status === "error"
                  ? "Error"
                  : "Idle"}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function AgentsPage() {
  const router = useRouter();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"grid" | "list">("grid");
  const [search, setSearch] = useState("");

  const fetchAgents = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agents: ${response.statusText}`);
      }
      const data = await response.json();
      setAgents(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch agents");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  // Filter agents by search
  const filteredAgents = useMemo(() => {
    if (!search) return agents;
    const searchLower = search.toLowerCase();
    return agents.filter(
      (agent) =>
        agent.role.toLowerCase().includes(searchLower) ||
        agent.goal.toLowerCase().includes(searchLower)
    );
  }, [agents, search]);

  // Group agents by crew
  const groupedAgents = useMemo(() => {
    return {
      staff: filteredAgents.filter((a) => a.crew === "staff"),
      spot: filteredAgents.filter((a) => a.crew === "spot"),
      futures: filteredAgents.filter((a) => a.crew === "futures"),
    };
  }, [filteredAgents]);

  const totalAgents = agents.length;
  const activeAgents = agents.filter((a) => a.status === "active").length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Loading agents...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertCircle className="h-12 w-12 text-red-500" />
          <div>
            <p className="font-medium">Failed to load agents</p>
            <p className="text-sm text-muted-foreground mt-1">{error}</p>
          </div>
          <Button onClick={fetchAgents} variant="outline" className="gap-2">
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
            Agents
          </h2>
          <p className="text-sm text-muted-foreground terminal-text">
            {activeAgents} active of {totalAgents} agents
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={() => router.push("/agents/create")}>
            <Plus className="h-4 w-4 mr-2" />
            Create Agent
          </Button>
          <Button
            onClick={fetchAgents}
            variant="ghost"
            size="icon"
            className="h-9 w-9"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search agents..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-64 pl-8"
            />
          </div>
          <div className="flex items-center border rounded-lg">
            <Button
              variant={view === "grid" ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setView("grid")}
              className="rounded-r-none"
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={view === "list" ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setView("list")}
              className="rounded-l-none"
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Tabs by Desk */}
      <Tabs defaultValue="all" className="space-y-4">
        <TabsList>
          <TabsTrigger value="all">All ({filteredAgents.length})</TabsTrigger>
          <TabsTrigger value="staff" className="gap-1.5">
            <div className="h-2 w-2 rounded-full bg-purple-500" />
            STAFF ({groupedAgents.staff.length})
          </TabsTrigger>
          <TabsTrigger value="spot" className="gap-1.5">
            <div className="h-2 w-2 rounded-full bg-orange-500" />
            Spot ({groupedAgents.spot.length})
          </TabsTrigger>
          <TabsTrigger value="futures" className="gap-1.5">
            <div className="h-2 w-2 rounded-full bg-teal-500" />
            Futures ({groupedAgents.futures.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-6">
          {/* STAFF */}
          {groupedAgents.staff.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-purple-500" />
                STAFF
              </h3>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {groupedAgents.staff.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={agent}
                    onClick={() => router.push(`/agents/${agent.id}`)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Spot */}
          {groupedAgents.spot.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-orange-500" />
                Spot Desk
              </h3>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {groupedAgents.spot.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={agent}
                    onClick={() => router.push(`/agents/${agent.id}`)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Futures */}
          {groupedAgents.futures.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-teal-500" />
                Futures Desk
              </h3>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {groupedAgents.futures.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={agent}
                    onClick={() => router.push(`/agents/${agent.id}`)}
                  />
                ))}
              </div>
            </div>
          )}

          {filteredAgents.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              No agents found matching "{search}"
            </div>
          )}
        </TabsContent>

        <TabsContent value="staff">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {groupedAgents.staff.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onClick={() => router.push(`/agents/${agent.id}`)}
              />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="spot">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {groupedAgents.spot.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onClick={() => router.push(`/agents/${agent.id}`)}
              />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="futures">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {groupedAgents.futures.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onClick={() => router.push(`/agents/${agent.id}`)}
              />
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
