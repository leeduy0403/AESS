import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  isEditMode: "false",
};

export const isEditModeSlice = createSlice({
  name: "isEditMode",
  initialState,
  reducers: {
    toggleIsEditMode: (state) => {
      state.isEditMode = !state.isEditMode;
    },
  },
});

// Action creators are generated for each case reducer function
export const { toggleIsEditMode } = isEditModeSlice.actions;

export default isEditModeSlice.reducer;
