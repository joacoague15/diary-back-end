from django.http import HttpResponse


def home_view(request):
    return HttpResponse("Welcome to the Home Page!")


def about_view(request):
    return HttpResponse("This is the About Page.")


def contact_view(request):
    return HttpResponse("Contact us at contact@example.com.")
