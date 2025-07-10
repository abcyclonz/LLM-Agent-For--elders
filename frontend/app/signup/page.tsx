"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/components/auth-context"
import { toast } from "sonner"
import Image from "next/image"
import { useTheme } from "next-themes"
import { useEffect } from "react"

interface SignUpFormData {
  email: string
  password: string
  fullName: string
  age: string
  preferredLanguage: string
  background: string
  interests: string[]
  conversationPreferences: string[]
  technologyUsage: string
  conversationGoals: string[]
  additionalInfo: string
}

export default function SignUpPage() {
  const router = useRouter()
  const { signup } = useAuth()
  const [formData, setFormData] = useState<SignUpFormData>({
    email: "",
    password: "",
    fullName: "",
    age: "",
    preferredLanguage: "",
    background: "",
    interests: [],
    conversationPreferences: [],
    technologyUsage: "",
    conversationGoals: [],
    additionalInfo: ""
  })

  const [otherInterest, setOtherInterest] = useState("")
  const [otherGoal, setOtherGoal] = useState("")
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  const interests = [
    "History",
    "Watching old movies",
    "Woodworking",
    "Cricket",
    "Reading",
    "Gardening",
    "Travel",
    "Family stories",
    "Learning new things"
  ]

  const conversationPreferences = [
    "Respectful and thoughtful discussions",
    "Casual and friendly chats",
    "Engaging and chatty conversations",
    "Discussions where my experiences are acknowledged",
    "Informative discussions (e.g., getting news, facts)",
    "Light-hearted conversations and jokes"
  ]

  const conversationGoals = [
    "Discussing various topics of interest",
    "Sharing memories and stories from my past",
    "Just a friendly chat",
    "Getting help finding information online",
    "Having light-hearted conversations",
    "Feeling understood and less lonely"
  ]

  const languages = [
    "English",
    "Hindi",
    "Malayalam",
    "Tamil",
    "Telugu",
    "Kannada",
    "Bengali",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Other"
  ]

  const handleInterestChange = (interest: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      interests: checked 
        ? [...prev.interests, interest]
        : prev.interests.filter(i => i !== interest)
    }))
  }

  const handleConversationPreferenceChange = (preference: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      conversationPreferences: checked 
        ? [...prev.conversationPreferences, preference]
        : prev.conversationPreferences.filter(p => p !== preference)
    }))
  }

  const handleConversationGoalChange = (goal: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      conversationGoals: checked 
        ? [...prev.conversationGoals, goal]
        : prev.conversationGoals.filter(g => g !== goal)
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Add "Other" interests and goals if specified
    const finalInterests = [...formData.interests]
    if (otherInterest.trim()) {
      finalInterests.push(otherInterest.trim())
    }

    const finalGoals = [...formData.conversationGoals]
    if (otherGoal.trim()) {
      finalGoals.push(otherGoal.trim())
    }

    const submitData = {
      email: formData.email,
      password: formData.password,
      fullName: formData.fullName,
      age: formData.age,
      preferredLanguage: formData.preferredLanguage,
      background: formData.background,
      interests: finalInterests,
      conversationPreferences: formData.conversationPreferences,
      technologyUsage: formData.technologyUsage,
      conversationGoals: finalGoals,
      additionalInfo: formData.additionalInfo
    }

    try {
      await signup(submitData)
      toast.success("Account created successfully!")
      router.push("/login")
    } catch (error) {
      toast.error("Failed to create account. Please try again.")
    }
  }

  return (
    <div className="min-h-screen bg-background dark:from-gray-900 dark:to-gray-800 py-8 px-4">
      <div className="max-w-4xl mx-auto mt-14">
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
            Welcome to Memora
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Tell us a little about yourself so we can make our conversations more enjoyable and helpful for you.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Basic Information */}
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Your Basic Information</CardTitle>
              <CardDescription>
                Let's start with some basic details about you.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  placeholder="Enter your email address"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                  placeholder="Enter a password"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="fullName">Your Full Name</Label>
                <Input
                  id="fullName"
                  value={formData.fullName}
                  onChange={(e) => setFormData(prev => ({ ...prev, fullName: e.target.value }))}
                  placeholder="Enter your full name"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="age">Your Age</Label>
                <Input
                  id="age"
                  value={formData.age}
                  onChange={(e) => setFormData(prev => ({ ...prev, age: e.target.value }))}
                  placeholder="e.g., 70s, Early 80s, or just your age"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="language">What language do you prefer to chat in?</Label>
                <Select
                  value={formData.preferredLanguage}
                  onValueChange={(value) => setFormData(prev => ({ ...prev, preferredLanguage: value }))}
                  required
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select your preferred language" />
                  </SelectTrigger>
                  <SelectContent>
                    {languages.map((language) => (
                      <SelectItem key={language} value={language}>
                        {language}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Story & Interests */}
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Your Story & Interests</CardTitle>
              <CardDescription>
                Share a bit about your background and what interests you.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="background">Could you tell us a little about your background or what you used to do?</Label>
                <Textarea
                  id="background"
                  value={formData.background}
                  onChange={(e) => setFormData(prev => ({ ...prev, background: e.target.value }))}
                  placeholder="e.g., I was a history teacher for many years..."
                  className="min-h-[100px]"
                  required
                />
              </div>

              <div className="space-y-4">
                <Label>What are some of your favorite things to do or talk about?</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {interests.map((interest) => (
                    <div key={interest} className="flex items-center space-x-2">
                      <Checkbox
                        id={interest}
                        checked={formData.interests.includes(interest)}
                        onCheckedChange={(checked) => handleInterestChange(interest, checked as boolean)}
                      />
                      <Label htmlFor={interest} className="text-sm font-normal">
                        {interest}
                      </Label>
                    </div>
                  ))}
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="other-interest"
                    checked={otherInterest.trim() !== ""}
                    onCheckedChange={(checked) => {
                      if (!checked) setOtherInterest("")
                    }}
                  />
                  <Label htmlFor="other-interest" className="text-sm font-normal">
                    Other:
                  </Label>
                  <Input
                    value={otherInterest}
                    onChange={(e) => setOtherInterest(e.target.value)}
                    placeholder="Specify your interest"
                    className="flex-1"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* How We Can Best Connect */}
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">How We Can Best Connect With You</CardTitle>
              <CardDescription>
                Help us understand how to make our conversations most meaningful for you.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <Label>What kind of conversations do you enjoy most?</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {conversationPreferences.map((preference) => (
                    <div key={preference} className="flex items-center space-x-2">
                      <Checkbox
                        id={preference}
                        checked={formData.conversationPreferences.includes(preference)}
                        onCheckedChange={(checked) => handleConversationPreferenceChange(preference, checked as boolean)}
                      />
                      <Label htmlFor={preference} className="text-sm font-normal">
                        {preference}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="technology">What kind of technology do you typically use?</Label>
                <Input
                  id="technology"
                  value={formData.technologyUsage}
                  onChange={(e) => setFormData(prev => ({ ...prev, technologyUsage: e.target.value }))}
                  placeholder="e.g., I use a tablet for news, A smartphone for calls"
                  required
                />
              </div>

              <div className="space-y-4">
                <Label>What would you like our conversations to be about, or what kind of help are you looking for?</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {conversationGoals.map((goal) => (
                    <div key={goal} className="flex items-center space-x-2">
                      <Checkbox
                        id={goal}
                        checked={formData.conversationGoals.includes(goal)}
                        onCheckedChange={(checked) => handleConversationGoalChange(goal, checked as boolean)}
                      />
                      <Label htmlFor={goal} className="text-sm font-normal">
                        {goal}
                      </Label>
                    </div>
                  ))}
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="other-goal"
                    checked={otherGoal.trim() !== ""}
                    onCheckedChange={(checked) => {
                      if (!checked) setOtherGoal("")
                    }}
                  />
                  <Label htmlFor="other-goal" className="text-sm font-normal">
                    Something else:
                  </Label>
                  <Input
                    value={otherGoal}
                    onChange={(e) => setOtherGoal(e.target.value)}
                    placeholder="Specify your goal"
                    className="flex-1"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Anything Else */}
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Anything Else?</CardTitle>
              <CardDescription>
                Is there anything else you'd like us to know about you?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label htmlFor="additionalInfo">Additional Information</Label>
                <Textarea
                  id="additionalInfo"
                  value={formData.additionalInfo}
                  onChange={(e) => setFormData(prev => ({ ...prev, additionalInfo: e.target.value }))}
                  placeholder="Anything else you'd like us to know about you that would help us be a better friend or companion..."
                  className="min-h-[100px]"
                />
              </div>
            </CardContent>
          </Card>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button type="submit" size="lg" className="flex-1 sm:flex-none">
              Create Account
            </Button>
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={() => router.push("/login")}
              className="flex-1 sm:flex-none"
            >
              Already have an account? Sign in
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
} 