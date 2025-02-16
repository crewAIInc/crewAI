import CrewDetail from "@/components/CrewDetail";
import { Metadata } from "next";

interface PageParams {
  params: {
    id: string;
  };
}

export default function Page({ params }: PageParams) {
  return (
    <main className="container mx-auto px-4 py-8">
      <CrewDetail />
    </main>
  );
}

export const metadata: Metadata = {
  title: "Crew Details | Soln.ai",
  description: "View detailed information about a specific crew and its agents",
};
