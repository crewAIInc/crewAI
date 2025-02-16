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
import { useStore } from "@/lib/store";
import type { Agent } from "@/lib/types";
import Link from "next/link";
import { useEffect } from "react";

interface AgentCardProps {
  agent: Agent;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{agent.name}</CardTitle>
        <CardDescription>{agent.role}</CardDescription>
      </CardHeader>
      <CardContent>
        <Badge
          variant={agent.agent_type === "solnai" ? "default" : "secondary"}
        >
          {agent.agent_type}
        </Badge>
        {agent.agent_type === "autogen" && agent.autogen_config && (
          <div className="mt-2">
            <p className="text-sm text-muted-foreground">Model</p>
            <p>{agent.autogen_config.llm_config?.model}</p>
          </div>
        )}
        {agent.agent_type === "solnai" && agent.llm && (
          <div className="mt-2">
            <p className="text-sm text-muted-foreground">LLM</p>
            <p>{agent.llm}</p>
          </div>
        )}
      </CardContent>
      <CardFooter>
        <Button asChild variant="link">
          <Link href={`/agents/${agent.id}`}>View Details</Link>
        </Button>
      </CardFooter>
    </Card>
  );
};

const AgentList = () => {
  const { agents, fetchAgents, isLoading, error } = useStore();

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <CardTitle>
                <Skeleton className="h-6 w-2/3" />
              </CardTitle>
              <CardDescription>
                <Skeleton className="h-4 w-1/2" />
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Skeleton className="h-4 w-full" />
            </CardContent>
            <CardFooter>
              <Skeleton className="h-8 w-20" />
            </CardFooter>
          </Card>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-red-500">Error</CardTitle>
        </CardHeader>
        <CardContent>
          <div role="alert" className="text-red-500">
            Error:{" "}
            {error instanceof Error
              ? error.message
              : "An unknown error occurred"}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!agents.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Agents Found</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            There are no agents in the system yet. You can add agents when
            creating a new crew.
          </p>
          <Link
            href="/crews/new"
            className="text-blue-500 hover:underline mt-4 inline-block"
          >
            Create a New Crew
          </Link>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {agents.map((agent) => (
        <AgentCard key={agent.id} agent={agent} />
      ))}
    </div>
  );
};

export default AgentList;
