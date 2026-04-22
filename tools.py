def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    print(f"\n{'='*50}")
    print(f"  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*50}\n")
    return {
        "status": "success",
        "message": f"Lead captured successfully: {name}, {email}, {platform}",
        "data": {
            "name": name,
            "email": email,
            "platform": platform
        }
    }
