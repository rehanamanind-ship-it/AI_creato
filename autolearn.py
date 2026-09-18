class AutoLearn:
    """Small local auto-selection helper for AI Creator training modes."""

    def choose_model(self, training_rows):
        example_count = len(training_rows)
        average_output_length = self.average_output_length(training_rows)

        if example_count < 100:
            return "SLM"
        if average_output_length <= 3:
            return "MLM"
        return "LLM"

    @staticmethod
    def average_output_length(training_rows):
        if not training_rows:
            return 0
        total_words = 0
        for row in training_rows:
            output = row.get("output", row.get("assistant", ""))
            total_words += len(output.split())
        return total_words / len(training_rows)
