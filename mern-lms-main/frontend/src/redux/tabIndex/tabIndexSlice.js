import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  tabIndex: 0,
};

export const tabIndexSlice = createSlice({
  name: "tabIndex",
  initialState,
  reducers: {
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },
  },
});

// Action creators are generated for each case reducer function
export const { setTabIndex } = tabIndexSlice.actions;

export default tabIndexSlice.reducer;
