#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	FILE *fp;
	unsigned char value;

	fp = fopen(argv[1], "rb+");

	while (!feof(fp)) {
		fscanf(fp, "%c", &value);

		if (!feof(fp)) {
			printf("%d\n", value);
		}
	}

	return 0;
}
