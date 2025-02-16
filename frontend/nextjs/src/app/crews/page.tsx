import CrewList from "@/components/CrewList";
import { Button } from "@/components/ui/button";
import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Crews | Soln.ai",
  description: "Manage your crews and agents",
};

export default function CrewsPage() {
  return (
    <main className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Crews</h1>
        <Link href="/crews/new">
          <Button>Create Crew</Button>
        </Link>
      </div>
      <CrewList />
    </main>
  );
}
