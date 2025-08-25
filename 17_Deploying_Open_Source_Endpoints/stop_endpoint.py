#!/usr/bin/env python3
"""
Emergency script to stop dedicated endpoints on Together AI
Run this immediately to stop your running endpoints!
"""

import os
from together import Together
import dotenv

# Load environment variables
dotenv.load_dotenv()

def stop_dedicated_endpoints():
    """Stop all running dedicated endpoints"""
    try:
        # Initialize Together client
        client = Together()
        
        # List all endpoints
        print("Fetching your endpoints...")
        endpoints = client.endpoints.list()
        
        running_endpoints = []
        for endpoint in endpoints:
            if hasattr(endpoint, 'status') and endpoint.status == 'running':
                running_endpoints.append(endpoint)
                print(f"Found running endpoint: {endpoint.name} (ID: {endpoint.id})")
        
        if not running_endpoints:
            print("✅ No running dedicated endpoints found!")
            return
        
        # Stop each running endpoint
        for endpoint in running_endpoints:
            print(f"🛑 Stopping endpoint: {endpoint.name}")
            try:
                # Stop the endpoint
                client.endpoints.delete(endpoint.id)
                print(f"✅ Successfully stopped: {endpoint.name}")
            except Exception as e:
                print(f"❌ Failed to stop {endpoint.name}: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing Together AI: {e}")
        print("Please check your API key and internet connection")

if __name__ == "__main__":
    print("🚨 EMERGENCY ENDPOINT SHUTDOWN 🚨")
    print("Stopping all running dedicated endpoints...")
    stop_dedicated_endpoints()
    print("\n💡 Remember to always set auto-shutdown timers for future endpoints!")
