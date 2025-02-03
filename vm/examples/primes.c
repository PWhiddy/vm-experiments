
void main() {
    for (int n = 2; n < 255; n = n + 1) {
        int is_prime = 1;
        int max = n + 1;
        for (int i = 2; i * i < max; i = i + 1) {
            int md = n % i;
            int cmp = md == 0;
            if (cmp) {
                is_prime = 0;
                break;
            }
        }
        if (is_prime) {
            printf(n);
        }
    }
}