function Calendar() {
  return (
    <div className="relative flex items-center justify-center py-10 bg-[#E4E4E4]">
      <div className="border-t-black absolute w-10 h-10 border-4 border-t-4 border-gray-300 rounded-full animate-spin"></div>
      <iframe
        className="w-[80vw] h-[80vh] border-2 border-black rounded-md"
        src="https://calendar.google.com/calendar/embed?src=triet.nguyenminhbk2908%40hcmut.edu.vn&ctz=Asia%2FHo_Chi_Minh"
        onLoad={(e) => e.target.previousSibling.remove()}
      ></iframe>
    </div>
  );
}

export default Calendar;
