import AgentDetail from "@/components/AgentDetail";
import { Metadata } from "next";

interface PageParams {
  params: {
    id: string;
  };
}

export const metadata: Metadata = {
  title: "Agent Detail | Soln.ai",
  description: "View agent details",
};

export default function AgentDetailPage({ params }: PageParams) {
  return (
    <main className="container mx-auto p-4">
      <AgentDetail agentId={params.id} />
    </main>
  );
}
