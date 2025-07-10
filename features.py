import re

suspicious_keywords = ['login', 'secure', 'account', 'update', 'free', 'bank', 'click', 'verify', 'signin', 'admin']

def extract_features(url):
    features = []

    # URL length
    features.append(len(url))

    # Number of dots
    features.append(url.count('.'))

    # Number of hyphens
    features.append(url.count('-'))

    # Number of slashes
    features.append(url.count('/'))

    # Number of subdomains
    subdomains = url.split('//')[-1].split('/')[0].split('.')
    features.append(max(0, len(subdomains) - 2))  # ignoring domain + TLD

    # Presence of IP address
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    features.append(1 if ip_pattern.search(url) else 0)

    # Presence of '@' symbol
    features.append(1 if '@' in url else 0)

    # Presence of '//' after protocol (redirect)
    features.append(1 if url.find('//', 8) != -1 else 0)

    # HTTPS presence (1 if https, 0 otherwise)
    features.append(1 if url.startswith('https://') else 0)

    # Count suspicious keywords
    url_lower = url.lower()
    count_suspicious = sum(keyword in url_lower for keyword in suspicious_keywords)
    features.append(count_suspicious)

    # Count digits
    features.append(sum(c.isdigit() for c in url))

    # Count special characters (excluding alphanumeric)
    features.append(sum(not c.isalnum() for c in url))

    return features
