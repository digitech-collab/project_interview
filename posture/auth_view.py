from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages

def user_login(request):
    """Handle user login."""
    if request.user.is_authenticated:
        return redirect('index')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            # Redirect to a success page
            return redirect('index')
        else:
            # Return an 'invalid login' error message
            return render(request, 'posture_detection/login.html', {
                'error_message': 'Invalid username or password.'
            })
    
    return render(request, 'posture_detection/login.html')

def user_signup(request):
    """Handle user registration."""
    if request.user.is_authenticated:
        return redirect('index')
        
    if request.method == 'POST':
        # Get form data
        username = request.POST.get('username')
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Validate form data
        if password1 != password2:
            return render(request, 'posture_detection/signup.html', {
                'error_message': 'Passwords do not match.'
            })
        
        # Check if username or email already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'posture_detection/signup.html', {
                'error_message': 'Username already exists.'
            })
            
        if User.objects.filter(email=email).exists():
            return render(request, 'posture_detection/signup.html', {
                'error_message': 'Email already in use.'
            })
        
        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1,
                first_name=first_name,
                last_name=last_name
            )
            
            # Log the user in
            login(request, user)
            
            # Redirect to a success page
            return redirect('index')
            
        except Exception as e:
            return render(request, 'posture_detection/signup.html', {
                'error_message': str(e)
            })
    
    return render(request, 'posture_detection/signup.html')

@login_required
def user_logout(request):
    """Handle user logout."""
    logout(request)
    return redirect('login')

@login_required
def profile(request):
    """Display user profile."""
    return render(request, 'posture_detection/profile.html', {
        'user': request.user
    })