import mongoose from "mongoose";

const questionSchema = new mongoose.Schema(
  {
    content: {
      type: String,
      required: true,
    },
    replies: {
      type: Array,
      default: [],
    },
    userId: {
      type: String,
      required: true,
    },
    topicId: {
      type: String,
      required: true,
    },
    replies: {
      type: Array,
      default: [],
    },
  },
  { timestamps: true }
);

const Question = mongoose.model("Question", questionSchema);

export default Question;
