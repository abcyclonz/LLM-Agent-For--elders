"use client"

import { SeniorChat } from "@/components/senior-chat"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuth } from "@/components/auth-context"
import Link from "next/link"
import Image from "next/image" // Import Image component
import { useState, useEffect } from "react" // Import useState and useEffect
import { useTheme } from "next-themes" // Assuming you have a useTheme hook from next-themes or similar

export default function Page() {
  const { isAuthenticated } = useAuth()
  const [mounted, setMounted] = useState(false) // State for client-side mounting
  const { theme } = useTheme() // Get the current theme

  // Set mounted to true once the component mounts on the client side
  useEffect(() => {
    setMounted(true)
  }, [])

  if (isAuthenticated) {
    return <SeniorChat />
  }

  return (
    <div className="min-h-screen bg-background flex items-start justify-center px-2 pb-4">
      <div className="max-w-3xl mx-auto text-center mt-14">
        <div>
          {/* Smaller logo */}
          <div className="relative w-16 h-16 mx-auto mb-2 mt-8 flex-shrink-0">
            {mounted && (
              <Image
                src={theme === "dark" ? "/memora-dark.png" : "/memora-light.png"}
                alt="Memora"
                width={64}
                height={64}
                className="object-contain transition-smooth hover:scale-105"
                priority
              />
            )}
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome to Memora
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-6 max-w-xl mx-auto">
            Your kind, patient, and empathetic AI companion for memory assistance and meaningful conversations.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">For New Users</CardTitle>
              <CardDescription>
                Create your account and tell us about yourself so we can personalize your experience.
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <Link href="/signup">
                <Button size="sm" className="w-full">
                  Get Started
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Returning Users</CardTitle>
              <CardDescription>
                Welcome back! Sign in to continue your conversations with Memora.
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <Link href="/login">
                <Button variant="outline" size="sm" className="w-full">
                  Sign In
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        <div className="text-center mb-0 mt-10">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            What makes Memora special?
          </h2>
          <div className="grid md:grid-cols-3 gap-4 text-left">
            <div className="text-center">
              <div className="bg-blue-100 dark:bg-blue-900/20 w-9 h-9 rounded-full flex items-center justify-center mx-auto mb-2">
                <span className="text-xl">💬</span>
              </div>
              <h3 className="font-semibold mb-1">Patient Conversations</h3>
              <p className="text-gray-600 dark:text-gray-300 text-xs">
                Enjoy thoughtful, respectful discussions tailored to your interests and experiences.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 dark:bg-green-900/20 w-9 h-9 rounded-full flex items-center justify-center mx-auto mb-2">
                <span className="text-xl">🧠</span>
              </div>
              <h3 className="font-semibold mb-1">Memory Assistance</h3>
              <p className="text-gray-600 dark:text-gray-300 text-xs">
                Get help with remembering important information and organizing your thoughts.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 dark:bg-purple-900/20 w-9 h-9 rounded-full flex items-center justify-center mx-auto mb-2">
                <span className="text-xl">🤝</span>
              </div>
              <h3 className="font-semibold mb-1">Companionship</h3>
              <p className="text-gray-600 dark:text-gray-300 text-xs">
                Feel understood and less lonely with our empathetic AI companion.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}