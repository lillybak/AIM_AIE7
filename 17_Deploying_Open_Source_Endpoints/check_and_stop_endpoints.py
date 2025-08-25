#!/usr/bin/env python3
"""
Check credit usage and stop dedicated endpoints on Together AI
"""

import os
from together import Together
import dotenv

# Load environment variables
dotenv.load_dotenv()

def check_credits_and_stop_endpoints():
    """Check credit usage and stop running endpoints"""
    try:
        # Initialize Together client
        client = Together()
        
        # Check credit usage
        print("💰 Checking your credit usage...")
        try:
            # Try to get account info
            account_info = client.account.get()
            print(f"Account info: {account_info}")
        except Exception as e:
            print(f"Could not get account info: {e}")
        
        # List all endpoints
        print("\n🔍 Checking your endpoints...")
        try:
            endpoints = client.endpoints.list()
            
            running_endpoints = []
            for endpoint in endpoints:
                print(f"Endpoint: {endpoint.name} - Status: {getattr(endpoint, 'status', 'unknown')}")
                if hasattr(endpoint, 'status') and endpoint.status == 'running':
                    running_endpoints.append(endpoint)
            
            if not running_endpoints:
                print("✅ No running dedicated endpoints found!")
                return
            
            print(f"\n🚨 Found {len(running_endpoints)} running endpoint(s)!")
            
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
            print(f"❌ Error listing endpoints: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing Together AI: {e}")
        print("Please check your API key and internet connection")

if __name__ == "__main__":
    print("🚨 CREDIT USAGE CHECK & ENDPOINT SHUTDOWN 🚨")
    check_credits_and_stop_endpoints()
    print("\n💡 Remember to always set auto-shutdown timers for future endpoints!")
    print("💡 Consider using serverless endpoints for testing to avoid charges!")
