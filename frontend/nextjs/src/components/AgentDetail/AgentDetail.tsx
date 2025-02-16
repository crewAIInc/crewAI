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
import Link from "next/link";
import { useEffect } from "react";

interface AgentDetailProps {
  agentId: string;
}

const AgentDetail = ({ agentId }: AgentDetailProps) => {
  const { selectedAgent: agent, fetchAgent, isLoading, error } = useStore();

  useEffect(() => {
    fetchAgent(agentId);
  }, [fetchAgent, agentId]);

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
            Error:{" "}
            {error instanceof Error
              ? error.message
              : "An unknown error occurred"}
          </div>
        </CardContent>
        <CardFooter>
          <Button asChild variant="outline">
            <Link href="/agents">Back to Agents</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  if (!agent) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Agent Not Found</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            The requested agent could not be found.
          </p>
        </CardContent>
        <CardFooter>
          <Button asChild variant="outline">
            <Link href="/agents">Back to Agents</Link>
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
            <CardTitle className="text-2xl">{agent.name}</CardTitle>
            <CardDescription>Agent ID: {agent.id}</CardDescription>
          </div>
          <Badge
            variant={agent.agent_type === "solnai" ? "default" : "secondary"}
            className="ml-4"
          >
            {agent.agent_type}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold mb-2">Details</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Role</p>
              <p>{agent.role}</p>
            </div>
            {agent.goal && (
              <div>
                <p className="text-sm text-muted-foreground">Goal</p>
                <p>{agent.goal}</p>
              </div>
            )}
            {agent.backstory && (
              <div>
                <p className="text-sm text-muted-foreground">Backstory</p>
                <p>{agent.backstory}</p>
              </div>
            )}
            {agent.agent_type === "solnai" && agent.llm && (
              <div>
                <p className="text-sm text-muted-foreground">LLM</p>
                <p>{agent.llm}</p>
              </div>
            )}
            {agent.agent_type === "autogen" && agent.autogen_config && (
              <div>
                <p className="text-sm text-muted-foreground">Model</p>
                <p>{agent.autogen_config.llm_config?.model}</p>
              </div>
            )}
            <div>
              <p className="text-sm text-muted-foreground">Created</p>
              <p>{new Date(agent.created_at).toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Last Updated</p>
              <p>{new Date(agent.updated_at).toLocaleString()}</p>
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button asChild variant="outline">
          <Link href="/agents">Back to Agents</Link>
        </Button>
      </CardFooter>
    </Card>
  );
};

export default AgentDetail;
