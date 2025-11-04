import mongoose from "mongoose";

const gradeSchema = new mongoose.Schema(
  {
    courseId: {
      type: String,
      required: true,
    },
    userId: {
      type: String,
      required: true,
    },
    grade: {
      type: Array,
      default: [],
    },
  },
  { timestamps: true }
);

const Grade = mongoose.model("Grade", gradeSchema);

export default Grade;
