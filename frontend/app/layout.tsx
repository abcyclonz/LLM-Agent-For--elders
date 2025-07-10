import './globals.css'
import { cn } from "@/lib/utils"
import { TooltipProvider } from "@/components/ui/tooltip"
import { ThemeProvider } from "@/components/theme-provider"
import { AuthProvider } from "@/components/auth-context"
import { Inter } from "next/font/google"
import type { ReactNode } from "react"
import { ClientNavWrapper } from "@/components/ClientNavWrapper"
import { BodyPaddingManager } from "@/components/BodyPaddingManager";

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Memora",
  description: "Your kind, patient, and empathetic AI companion for memory assistance and conversation.",
  generator: 'v0.dev'
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" type="image/png" href="/memora-favicon.png" />
      </head>
      <body className={cn("flex min-h-svh flex-col antialiased", inter.className)}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange={false}>
          <AuthProvider>
            <TooltipProvider delayDuration={0}>
              <ClientNavWrapper />
              <BodyPaddingManager />
              {children}
            </TooltipProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}