#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	FILE *fp, *fpout;
	int value;

	fp = fopen(argv[1], "r+");
	fpout = fopen("output.raw", "wb+");

	while (!feof(fp)) {
		fscanf(fp, "%d", &value);
		if (!feof(fp)) {
			// printf("%d\n", value);
			printf("%c\n", value);
			fprintf(fpout, "%c", value);
		}
	}

	fclose(fp);
	fclose(fpout);

	return 0;
}
