import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cervical Cell Analysis",
  description: "Computational classification of cervical cells — research prototype.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
