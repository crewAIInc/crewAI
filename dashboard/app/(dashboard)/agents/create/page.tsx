"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { AgentForm } from "@/components/agents/agent-form";

export default function CreateAgentPage() {
  const router = useRouter();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={() => router.push("/agents")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="text-lg font-semibold">Create New Agent</h2>
      </div>
      <AgentForm
        onSuccess={(agent) => {
          router.push(`/agents/${agent.id}`);
        }}
        onCancel={() => router.push("/agents")}
      />
    </div>
  );
}
