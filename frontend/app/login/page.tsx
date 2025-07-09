"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuth } from "@/components/auth-context"
import { toast } from "sonner"
import { Eye, EyeOff, Loader2 } from "lucide-react" // Assuming you have lucide-react installed
import Image from "next/image"
import { useTheme } from "next-themes"

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()
  const [formData, setFormData] = useState({
    email: "",
    password: ""
  })
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false) // New state for password visibility
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setErrorMessage("");

    try {
      await login(formData.email, formData.password)
      toast.success("Welcome back!")
      router.push("/")
    } catch (error) {
      console.error("Login attempt failed:", error);
      if (error instanceof Error && error.message.includes("Invalid email or password")) {
        setErrorMessage("User not found");
        toast.error("User not found");
      } else {
        setErrorMessage("Login failed. Please check your credentials.");
        toast.error("Login failed. Please check your credentials.");
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background dark:from-gray-900 dark:to-gray-800 flex items-center justify-center py-2 px-4">
      <div className="w-full max-w-md mt-14">
        <div className="text-center mb-8">
          {mounted && (
            <Image
              src={theme === "dark" ? "/memora-dark.png" : "/memora-light.png"}
              alt="Memora"
              width={64}
              height={64}
              className="block object-contain transition-smooth hover:scale-105 mx-auto mb-4"
              priority
            />
          )}
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome Back
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Sign in to continue your conversations with Memora
          </p>
        </div>

        <Card className="shadow-lg"> {/* Slightly more prominent shadow */}
          <CardHeader>
            <CardTitle className="text-2xl text-center">Sign In</CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            {errorMessage && (
              <div className="text-red-500 text-sm text-center mb-2">{errorMessage}</div>
            )}
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  placeholder="Enter your email address"
                  required
                  disabled={isLoading}
                />
              </div>

              <div className="space-y-2 relative"> {/* Added 'relative' for password toggle */}
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"} // Dynamic type
                  value={formData.password}
                  onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                  placeholder="Enter your password"
                  required
                  disabled={isLoading}
                  className="pr-10" // Add padding for the icon
                />
                <Button
                  type="button" // Important: not submit
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-[30px] -translate-y-1/2 h-full px-2 text-gray-500 hover:bg-transparent" // Adjust top to align with input
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  <span className="sr-only">{showPassword ? "Hide password" : "Show password"}</span>
                </Button>
              </div>

              <Button
                type="submit"
                className="w-full"
                size="lg"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Signing in...
                  </>
                ) : (
                  "Sign In"
                )}
              </Button>

              {/* Optional: Add social logins here */}
              {/*
              <div className="relative my-4">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t border-gray-200 dark:border-gray-700" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                </div>
              </div>
              <Button variant="outline" className="w-full flex items-center justify-center gap-2" size="lg">
                <img src="/google-icon.svg" alt="Google" className="h-5 w-5" />
                Sign In with Google
              </Button>
              */}

              <div className="text-center mt-6"> {/* Adjusted margin-top */}
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Don't have an account?{" "}
                  <Link
                    href="/signup"
                    className="text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
                  >
                    Sign up here
                  </Link>
                </p>
              </div>

              <div className="text-center mt-4"> {/* Adjusted margin-top */}
                <Link
                  href="/forgot-password"
                  className="text-sm text-gray-600 hover:text-gray-500 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  Forgot your password?
                </Link>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}