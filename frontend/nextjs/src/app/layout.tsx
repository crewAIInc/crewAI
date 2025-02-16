import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Soln.ai",
  description: "AI-powered crew management",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
