import mongoose from "mongoose";

const courseSchema = new mongoose.Schema(
  {
    startAcademicYear: {
      type: Number,
      required: true,
    },
    endAcademicYear: {
      type: Number,
      required: true,
    },
    subjectId: {
      type: String,
      required: true,
    },
  },
  { timestamps: true }
);

const Course = mongoose.model("Course", courseSchema);

export default Course;
