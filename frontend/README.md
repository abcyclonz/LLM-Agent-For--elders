# Memora Frontend

A Next.js application providing a user-friendly interface for elder users to interact with an AI companion.

## Features

### Authentication System
- **Sign Up Page**: Comprehensive form for new users to provide detailed information about themselves
- **Login Page**: Simple authentication for returning users
- **User Context**: Manages authentication state across the application

### Sign Up Form Sections

1. **Basic Information**
   - Full Name
   - Age (flexible input format)
   - Preferred Language (dropdown with multiple Indian languages)

2. **Story & Interests**
   - Background/Profession (text area)
   - Interests (checkboxes with "Other" option)
   - Favorite activities and topics

3. **Connection Preferences**
   - Conversation style preferences
   - Technology usage description
   - Conversation goals and objectives

4. **Additional Information**
   - Open text area for any other relevant information

### User Experience Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Accessibility**: Large text, clear labels, and easy navigation
- **Dark/Light Theme**: Toggle between themes
- **Form Validation**: Required fields and user-friendly error messages
- **Loading States**: Visual feedback during form submission

## Pages

### `/` - Home Page
- Welcome screen for unauthenticated users
- Directs to signup or login
- Shows chat interface for authenticated users

### `/signup` - Sign Up Page
- Comprehensive form with all user information fields
- Organized in logical sections
- Responsive grid layout for checkboxes

### `/login` - Login Page
- Simple email/password authentication
- Links to signup and forgot password

## Components

### `AuthProvider`
- Manages user authentication state
- Provides login, logout, and signup functions
- Stores user data in localStorage

### `Navigation`
- Responsive navigation bar
- Shows different options based on authentication status
- Includes theme toggle

### Form Components
- Uses shadcn/ui components for consistent styling
- Form validation and error handling
- Accessible form controls

## API Routes

### `/api/auth/signup`
- Handles user registration
- Validates required fields
- Returns user data (ready for backend integration)

### `/api/auth/login`
- Handles user authentication
- Validates credentials
- Returns user data (ready for backend integration)

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Technology Stack

- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Component library
- **React Hook Form**: Form handling
- **Zod**: Schema validation
- **Sonner**: Toast notifications

## Future Enhancements

- Backend integration for persistent user data
- Password hashing and JWT authentication
- User profile management
- Password reset functionality
- Email verification
- Multi-language support
- Voice input capabilities
- Accessibility improvements

## File Structure

```
frontend/
├── app/
│   ├── api/auth/
│   │   ├── login/route.ts
│   │   └── signup/route.ts
│   ├── login/page.tsx
│   ├── signup/page.tsx
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── auth-context.tsx
│   ├── navigation.tsx
│   ├── senior-chat.tsx
│   └── ui/
└── README.md
``` 