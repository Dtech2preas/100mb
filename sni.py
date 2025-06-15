import socket
import time
import subprocess
import requests
from colorama import Fore, init
from urllib.parse import urlparse

# Initialize colorama
init(autoreset=True)

def print_banner():
    print(Fore.CYAN + "D-TECH SNI Host Checker")
    print(Fore.CYAN + "=========================\n")

def resolve_ip(host):
    print(Fore.YELLOW + "[*] Resolving IP address...")
    try:
        ip = socket.gethostbyname(host)
        print(Fore.CYAN + f"[+] IP Address: {ip}")
        return ip
    except Exception as e:
        print(Fore.RED + f"[✘] DNS resolution failed: {e}")
        return None

def ping_host(host):
    print(Fore.YELLOW + "[*] Pinging host...")
    try:
        result = subprocess.run(["ping", "-c", "3", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(Fore.GREEN + "[✔] Host is reachable via ICMP ping")
            return True
        else:
            print(Fore.RED + "[✘] Ping failed (host not reachable or ICMP blocked)")
            return False
    except Exception as e:
        print(Fore.RED + f"[✘] Ping error: {e}")
        return False

def test_tcp_connection(host):
    print(Fore.YELLOW + "[*] Testing TCP port 443...")
    try:
        start_time = time.time()
        socket.setdefaulttimeout(5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, 443))
        tcp_time = round(time.time() - start_time, 2)
        print(Fore.GREEN + f"[✔] TCP connection successful ({tcp_time}s)")
        return True, tcp_time
    except Exception as e:
        print(Fore.RED + f"[✘] TCP connection failed: {e}")
        return False, None

def test_https_request(host):
    print(Fore.YELLOW + "[*] Sending HTTPS GET request...")
    try:
        start_time = time.time()
        response = requests.get(f'https://{host}', timeout=5)
        http_time = round(time.time() - start_time, 2)
        if response.status_code == 200:
            print(Fore.GREEN + f"[✔] HTTPS request successful ({http_time}s)")
            return True, http_time, len(response.content)
        else:
            print(Fore.RED + f"[✘] HTTPS failed: Status {response.status_code}")
            return False, None, 0
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"[✘] HTTPS request failed: {e}")
        return False, None, 0

def check_redirection(host):
    print(Fore.YELLOW + "[*] Checking for redirection (HTTP to HTTPS or domain change)...")
    try:
        original_url = f"http://{host}"
        response = requests.get(original_url, allow_redirects=True, timeout=5)
        final_url = response.url

        orig_host = urlparse(original_url).hostname
        final_host = urlparse(final_url).hostname

        if orig_host.lower() == final_host.lower():
            print(Fore.GREEN + "[✔] No harmful redirection detected (same domain)")
            return True
        else:
            print(Fore.RED + f"[✘] Redirected to different domain: {final_url}")
            return False
    except Exception as e:
        print(Fore.RED + f"[✘] Redirection check failed: {e}")
        return False

def test_sni_connection(host):
    print(Fore.MAGENTA + "\n[~] Starting full validation...\n")

    ip = resolve_ip(host)
    ping_ok = ping_host(host)
    tcp_ok, tcp_time = test_tcp_connection(host)
    https_ok, https_time, content_size = test_https_request(host)
    redirect_ok = check_redirection(host)

    if not (tcp_ok and https_ok and redirect_ok):
        print(Fore.RED + "\n[✘] One or more critical tests failed.")
        print(Fore.RED + "[!] This host is likely NOT zero-rated or usable for tunneling.")
        return

    total_time = (tcp_time or 0) + (https_time or 0)
    rating = max(0, round(100 - total_time * 10))

    print(Fore.CYAN + f"\n[✓] All tests passed.")
    print(Fore.CYAN + f"[+] Content received: {content_size} bytes")
    print(Fore.CYAN + f"[+] Estimated quality rating: {rating}%")

    if ping_ok and rating >= 70 and content_size > 1000:
        print(Fore.GREEN + "\n✅ Recommended for use in VPN tunneling apps (e.g., HTTP Injector, HA Tunnel)")
    else:
        print(Fore.YELLOW + "\n⚠️ May work, but not ideal for tunneling. Try another host.")

def main():
    print_banner()
    sni_host = input(Fore.CYAN + "[?] Enter the SNI host (e.g. example.com): ").strip()
    test_sni_connection(sni_host)

if __name__ == "__main__":
    main()