import TaskList from "@/components/TaskList";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tasks | Soln.ai",
  description: "Manage your tasks",
};

export default function TasksPage() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Tasks</h1>
      <TaskList />
    </main>
  );
}
