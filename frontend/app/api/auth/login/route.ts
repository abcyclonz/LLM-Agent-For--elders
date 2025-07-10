import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Validate required fields
    if (!body.email || !body.password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      )
    }

    // In a real application, you would:
    // 1. Check if user exists in database
    // 2. Verify password hash
    // 3. Generate JWT token
    // 4. Return user data and token

    // For now, we'll simulate a successful login
    const mockUser = {
      id: '1',
      fullName: 'John Doe',
      age: '75',
      preferredLanguage: 'English',
      background: 'Retired teacher',
      interests: ['History', 'Reading'],
      conversationPreferences: ['Respectful and thoughtful discussions'],
      technologyUsage: 'I use a tablet for news',
      conversationGoals: ['Just a friendly chat'],
      additionalInfo: 'I enjoy quiet conversations'
    }

    return NextResponse.json({
      success: true,
      user: mockUser,
      message: 'Login successful'
    })

  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
} 