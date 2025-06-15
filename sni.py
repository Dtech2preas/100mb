import socket
import time
import subprocess
import requests
import threading
import ssl
import datetime
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def print_banner():
    print(Fore.CYAN + "\nD-TECH ZERO-RATE SNI TEST")
    print(Fore.CYAN + "==========================\n")

def resolve_ip(host):
    print(Fore.YELLOW + "[*] Resolving IP addresses (A records)...")
    try:
        ips = socket.gethostbyname_ex(host)[2]
        for ip in ips:
            print(Fore.CYAN + f"[+] IP Address: {ip}")
        return True
    except Exception as e:
        print(Fore.RED + f"[✘] DNS resolution failed: {e}")
        return False

def resolve_ipv6(host):
    print(Fore.YELLOW + "[*] Resolving IPv6 addresses (AAAA records)...")
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET6)
        ipv6s = set()
        for info in infos:
            ipv6 = info[4][0]
            ipv6s.add(ipv6)
        if ipv6s:
            for ip6 in ipv6s:
                print(Fore.CYAN + f"[+] IPv6 Address: {ip6}")
            return True
        else:
            print(Fore.RED + "[✘] No IPv6 addresses found.")
            return False
    except Exception as e:
        print(Fore.RED + f"[✘] IPv6 resolution failed: {e}")
        return False

def ping_host(host):
    print(Fore.YELLOW + "[*] Pinging host (ICMP), 4 packets...")
    try:
        result = subprocess.run(["ping", "-c", "4", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            avg_line = next((l for l in lines if "rtt min/avg/max/mdev" in l), None)
            if avg_line:
                avg_time = avg_line.split('=')[1].split('/')[1]
                print(Fore.GREEN + f"[✔] Host is pingable. Avg latency: {avg_time} ms")
            else:
                print(Fore.GREEN + "[✔] Host is pingable.")
            return True
        else:
            print(Fore.RED + "[✘] Host not reachable via ping (ICMP may be blocked).")
            return False
    except Exception as e:
        print(Fore.RED + f"[✘] Ping failed: {e}")
        return False

def test_tcp_443(host):
    print(Fore.YELLOW + "[*] Testing TCP port 443 (VPN access check)...")
    try:
        socket.setdefaulttimeout(5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, 443))
        print(Fore.GREEN + "[✔] TCP port 443 is open.")
        return True
    except Exception as e:
        print(Fore.RED + f"[✘] TCP 443 connection failed: {e}")
        return False

def test_tcp_443_ipv6(host):
    print(Fore.YELLOW + "[*] Testing TCP port 443 on IPv6 (VPN access check)...")
    try:
        socket.setdefaulttimeout(5)
        infos = socket.getaddrinfo(host, 443, socket.AF_INET6, socket.SOCK_STREAM)
        for info in infos:
            with socket.socket(info[0], info[1], info[2]) as s:
                s.connect(info[4])
                print(Fore.GREEN + "[✔] TCP port 443 is open on IPv6.")
                return True
        print(Fore.RED + "[✘] TCP 443 connection failed on IPv6.")
        return False
    except Exception as e:
        print(Fore.RED + f"[✘] TCP 443 IPv6 connection failed: {e}")
        return False

def check_ssl_expiry(host):
    print(Fore.YELLOW + "[*] Checking SSL certificate expiry...")
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                expiry_date = datetime.datetime.strptime(cert['notAfter'], "%b %d %H:%M:%S %Y %Z")
                expiry_date = expiry_date.replace(tzinfo=datetime.timezone.utc)
                now = datetime.datetime.now(datetime.timezone.utc)
                days_left = (expiry_date - now).days
                print(Fore.GREEN + f"[✔] SSL Certificate valid, expires in {days_left} days.")
                return True
    except Exception as e:
        print(Fore.RED + f"[✘] SSL check failed: {e}")
        return False

def check_tls_versions(host):
    print(Fore.YELLOW + "[*] Checking TLS versions support (TLS 1.2 and 1.3)...")
    supported = []
    for version in [ssl.TLSVersion.TLSv1_3, ssl.TLSVersion.TLSv1_2]:
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.minimum_version = version
            context.maximum_version = version
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.create_connection((host, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    supported.append(str(version).split('.')[-1])
        except Exception:
            pass

    if supported:
        print(Fore.GREEN + f"[✔] Supported TLS versions: {', '.join(supported)}")
        return True
    else:
        print(Fore.RED + "[✘] No supported TLS versions found (TLS 1.2 or 1.3).")
        return False

def test_https(host):
    print(Fore.YELLOW + "[*] Testing HTTPS access and analyzing headers...")
    try:
        start = time.time()
        response = requests.get(f"https://{host}", headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*"
        }, timeout=5)
        elapsed = time.time() - start
        size = len(response.content)
        kbps = round((size / elapsed) / 1024, 2) if elapsed else 0

        if response.status_code == 200:
            print(Fore.GREEN + "[✔] HTTPS GET request successful.")
            print(Fore.CYAN + f"[+] Response size: {size} bytes")
            print(Fore.CYAN + f"[+] Estimated Speed: {kbps} KB/s")

            sec_headers = {
                "Strict-Transport-Security": "HSTS",
                "Content-Security-Policy": "CSP",
                "X-Frame-Options": "X-Frame-Options",
                "X-Content-Type-Options": "X-Content-Type-Options",
                "Referrer-Policy": "Referrer-Policy",
                "Permissions-Policy": "Permissions-Policy"
            }
            print(Fore.YELLOW + "[*] Checking HTTP security headers...")
            for header, name in sec_headers.items():
                if header in response.headers:
                    print(Fore.GREEN + f"[✔] {name} header present: {response.headers[header]}")
                else:
                    print(Fore.RED + f"[✘] {name} header missing")

            return True, size, kbps
        else:
            print(Fore.RED + f"[✘] HTTPS returned status code: {response.status_code}")
            return False, size, kbps
    except Exception as e:
        print(Fore.RED + f"[✘] HTTPS request failed: {e}")
        return False, 0, 0

def test_http_options(host):
    print(Fore.YELLOW + "[*] Testing HTTP OPTIONS method support...")
    try:
        response = requests.options(f"https://{host}", timeout=5)
        allowed = response.headers.get("Allow", "Not specified")
        print(Fore.CYAN + f"[+] Allowed HTTP methods: {allowed}")
        return True
    except Exception as e:
        print(Fore.RED + f"[✘] HTTP OPTIONS request failed: {e}")
        return False

def test_sni_host(host):
    print_banner()

    ip_ok = resolve_ip(host)
    ipv6_ok = resolve_ipv6(host)
    ping_ok = ping_host(host)
    tcp_ok = test_tcp_443(host)
    tcp6_ok = test_tcp_443_ipv6(host)
    ssl_ok = check_ssl_expiry(host)
    tls_ok = check_tls_versions(host)
    https_ok, size, speed = test_https(host)
    options_ok = test_http_options(host)

    score = 0
    if ip_ok: score += 15
    if ipv6_ok: score += 5
    if tcp_ok: score += 20
    if tcp6_ok: score += 5
    if ssl_ok: score += 10
    if tls_ok: score += 10
    if https_ok: score += 15
    if options_ok: score += 10
    if size > 1000: score += 5
    if speed > 25: score += 5

    print(Fore.CYAN + "\n===== SUMMARY =====")
    print(Fore.YELLOW + f"Score: {score}/100")

    if score >= 70:
        print(Fore.GREEN + "✅ Zero-Rated Host: YES (usable for tunneling or free access)")
        with open("good_hosts.txt", "a") as f:
            f.write(host + "\n")
    else:
        print(Fore.RED + "❌ Zero-Rated Host: NO (may require data or is limited)")
    print(Fore.CYAN + "===================\n")

def main():
    host = input(Fore.CYAN + "[?] Enter the host (e.g. example.com): ").strip()
    test_sni_host(host)

if __name__ == "__main__":
    main()