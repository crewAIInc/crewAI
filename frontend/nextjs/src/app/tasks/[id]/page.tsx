import TaskDetail from "@/components/TaskDetail";
import { Metadata } from "next";

interface PageParams {
  params: {
    id: string;
  };
}

export const metadata: Metadata = {
  title: "Task Detail | Soln.ai",
  description: "View task details",
};

export default function TaskDetailPage({ params }: PageParams) {
  return (
    <main className="container mx-auto p-4">
      <TaskDetail taskId={params.id} />
    </main>
  );
}
