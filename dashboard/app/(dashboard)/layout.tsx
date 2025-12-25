import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { WebSocketProvider } from "@/components/providers/websocket-provider";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <WebSocketProvider>
      <div className="min-h-screen bg-background">
        {/* Sidebar */}
        <Sidebar />

        {/* Main Content */}
        <div className="pl-64">
          <Header />
          <main className="p-6">{children}</main>
        </div>
      </div>
    </WebSocketProvider>
  );
}
