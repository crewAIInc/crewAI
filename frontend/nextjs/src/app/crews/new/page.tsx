import CreateCrewForm from "@/components/CreateCrewForm";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Create Crew | Soln.ai",
  description: "Create a new crew with agents",
};

export default function CreateCrewPage() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Create New Crew</h1>
      <CreateCrewForm />
    </main>
  );
}
